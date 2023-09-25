#!/usr/bin/env bash

cd egs/gop_speechocean762/s5
bash run_prep.sh

gopt_path=/working
cd $gopt_path
mkdir -p tmp/raw_kaldi_gop
cp src/extract_kaldi_gop/extract_gop_feats.py ${gopt_path}/egs/gop_speechocean762/s5/local/
