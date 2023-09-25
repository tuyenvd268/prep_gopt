#!/usr/bin/env bash

start_time=`date +%s%N`

cd egs/gop_speechocean762/s5
bash run_prep.sh

cd ../../..
gopt_path=/working
cd $gopt_path
mkdir -p tmp/raw_kaldi_gop
cp src/extract_kaldi_gop/extract_gop_feats.py ${gopt_path}/egs/gop_speechocean762/s5/local/

cd ${gopt_path}/egs/gop_speechocean762/s5
KALDI_ROOT=$gopt_path python3 local/extract_gop_feats.py
cd $gopt_path
cp -r ${gopt_path}/egs/gop_speechocean762/s5/gopt_feats/* data/raw_kaldi_gop/librispeech
end_time=`date +%s%N`

cd src/prep_data
KALDI_ROOT=$gopt_path python3 gen_seq_data.py

echo Execution time was `expr $end_time - $start_time` nanoseconds .
