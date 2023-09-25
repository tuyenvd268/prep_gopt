sudo nvidia-docker run -it --gpus all -p 6666:6666 \
    -v /data/codes/prep_gopt:/working \
    -v /data/audio_data/prep_submission_audio/10:/wav_dir \
    prep/kaldi_gpu-cuda_12.0.1 
streamlit run run_demo.py --server.port=8051 --server.address=0.0.0.0