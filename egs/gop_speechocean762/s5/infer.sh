#!/usr/bin/env bash

# audio_path="/working/egs/gop_speechocean762/s5/data/speechocean762/WAVE/SPEAKER1030/010300105.WAV"
# text="THE PAPER PUBLISHED NO DETAILS AND THERE WAS NO ATTACK"
path=$1
text=$2

rm -r /working/egs/gop_speechocean762/s5/data/infer/*

echo "TEXT: $text";
echo "PATH: $path";

python3 prepare_data.py --path "$path" --text "$text"

stage=1
nj=1

. ./cmd.sh
. ./path.sh
. parse_options.sh

librispeech_eg=../../librispeech/s5
model=$librispeech_eg/exp/chain_cleaned/tdnn_1d_sp
ivector_extractor=$librispeech_eg/exp/nnet3_cleaned/extractor
lang=$librispeech_eg/data/lang

for d in $model $ivector_extractor $lang; do
    [ ! -d $d ] && echo "$0: no such path $d" && exit 1;
done

# if [ $stage -le 2 ]; then
#   # Prepare data
#   for part in temp; do
#     local/data_prep.sh $data/speechocean762/$part data/$part
#   done

#   mkdir -p data/local
#   cp $data/speechocean762/resource/* data/local
# fi

if [ $stage -le 3 ]; then
  # Create high-resolution MFCC features
  for part in infer; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$cmd" data/$part || exit 1;
    steps/compute_cmvn_stats.sh data/$part || exit 1;
    utils/fix_data_dir.sh data/$part
  done
fi

if [ $stage -le 4 ]; then
  # Extract ivector
  for part in infer; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj $nj \
      data/$part $ivector_extractor data/$part/ivectors || exit 1;
  done
fi

if [ $stage -le 5 ]; then
  # Compute Log-likelihoods
  for part in infer; do
    steps/nnet3/compute_output.sh --cmd "$cmd" --nj $nj \
      --online-ivector-dir data/$part/ivectors data/$part $model exp/probs_$part
  done
fi

if [ $stage -le 7 ]; then
  # Split data and make phone-level transcripts
  for part in infer; do
    utils/split_data.sh data/$part $nj
    for i in `seq 1 $nj`; do
      utils/sym2int.pl -f 2- data/lang_nosp/words.txt \
        data/$part/split${nj}/$i/text \
        > data/$part/split${nj}/$i/text.int
    done

    utils/sym2int.pl -f 2- data/lang_nosp/phones.txt \
      data/local/text-phone > data/local/text-phone.int
  done
fi

if [ $stage -le 8 ]; then
  # Make align graphs
  for part in infer; do
    $cmd JOB=1:$nj exp/ali_$part/log/mk_align_graph.JOB.log \
      compile-train-graphs-without-lexicon \
        --read-disambig-syms=data/lang_nosp/phones/disambig.int \
        $model/tree $model/final.mdl \
        "ark,t:data/$part/split${nj}/JOB/text.int" \
        "ark,t:data/local/text-phone.int" \
        "ark:|gzip -c > exp/ali_$part/fsts.JOB.gz"   || exit 1;
    echo $nj > exp/ali_$part/num_jobs
  done
fi

if [ $stage -le 9 ]; then
  # Align
  for part in infer; do
    steps/align_mapped.sh --cmd "$cmd" --nj $nj --graphs exp/ali_$part \
      data/$part exp/probs_$part $lang $model exp/ali_$part
  done
fi

if [ $stage -le 10 ]; then
  local/remove_phone_markers.pl $lang/phones.txt \
    data/lang_nosp/phones-pure.txt data/lang_nosp/phone-to-pure-phone.int
fi

if [ $stage -le 11 ]; then
  for part in infer; do
    $cmd JOB=1:$nj exp/ali_$part/log/ali_to_phones.JOB.log \
      ali-to-phones --per-frame=true $model/final.mdl \
        "ark,t:gunzip -c exp/ali_$part/ali.JOB.gz|" \
        "ark,t:|gzip -c >exp/ali_$part/ali-phone.JOB.gz"   || exit 1;
  done
fi

if [ $stage -le 12 ]; then
  for part in infer; do
    $cmd JOB=1:$nj exp/gop_$part/log/compute_gop.JOB.log \
      compute-gop --phone-map=data/lang_nosp/phone-to-pure-phone.int \
        --skip-phones-string=0:1:2 \
        $model/final.mdl \
        "ark,t:gunzip -c exp/ali_$part/ali-phone.JOB.gz|" \
        "ark:exp/probs_$part/output.JOB.ark" \
        "ark,scp:exp/gop_$part/gop.JOB.ark,exp/gop_$part/gop.JOB.scp" \
        "ark,scp:exp/gop_$part/feat.JOB.ark,exp/gop_$part/feat.JOB.scp"   || exit 1;
      cat exp/gop_$part/feat.*.scp > exp/gop_$part/feat.scp
      cat exp/gop_$part/gop.*.scp > exp/gop_$part/gop.scp
  done
fi
