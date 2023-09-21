sudo nvidia-docker run -it --gpus all -p 8501:8501 \
    -v /data/codes/prep_gopt:/working \
    prep/kaldi_gpu-cuda_12.0.1
