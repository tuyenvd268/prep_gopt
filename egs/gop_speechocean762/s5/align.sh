librispeech_eg=../../librispeech/s5
model=$librispeech_eg/exp/chain_cleaned/tdnn_1d_sp

for i in  exp/ali_infer/ali.*.gz;
do /opt/kaldi/src/bin/ali-to-phones --ctm-output $model/final.mdl  \
ark:"gunzip -c $i|" -> ${i%.gz}.ctm;
done;

cd exp/ali_infer
cat *.ctm > merged_alignment.txt

#  Created by Eleanor Chodroff on 3/24/15.
#
Rscript /working/egs/gop_speechocean762/s5/id2phone.R
