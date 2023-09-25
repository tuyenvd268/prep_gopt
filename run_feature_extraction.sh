#!/usr/bin/env bash
# echo "PATH: $1";
# echo "TEXT: $2";
start_time=`date +%s%N`

cd egs/gop_speechocean762/s5
bash infer.sh $1 "$2"
# bash infer.sh

cd ../../..
gopt_path=/working
cd $gopt_path
mkdir -p data/raw_kaldi_gop/librispeech
cp src/extract_kaldi_gop/extract_gop_feats_for_infer.py ${gopt_path}/egs/gop_speechocean762/s5/local/

cd ${gopt_path}/egs/gop_speechocean762/s5
KALDI_ROOT=$gopt_path python3 local/extract_gop_feats_for_infer.py
cd $gopt_path
cp -r ${gopt_path}/egs/gop_speechocean762/s5/gopt_feats/* data/raw_kaldi_gop/librispeech
end_time=`date +%s%N`

cd src/prep_data
KALDI_ROOT=$gopt_path python3 gen_seq_data_phn_for_infer.py

echo Execution time was `expr $end_time - $start_time` nanoseconds .
