from omegaconf import OmegaConf
from glob import glob 
from tqdm import tqdm
import pandas as pd
import shutil
import json
import os

from utils import (
    create_text_phoneme,
    prepare_data_in_kaldi_gop_format
)
from extract_gop_feats import extract_kaldi_gop_feature
from gen_seq_data import generate_sequence_data_for_score_model

if __name__ == "__main__":
    config = OmegaConf.load("data_config.yml")
    print(
        json.dumps(
            OmegaConf.to_container(config), 
            indent=4, ensure_ascii=False)
        )
    metadata = pd.read_csv(config.metadata_path, index_col=0)
    metadata["wav_path"] = metadata.id.apply(lambda x: os.path.join(config.wav_dir, f'{x}.wav'))
    metadata["text"] = metadata["question_content"]
    metadata["spk_id"] = metadata.id.apply(lambda x: x)
    
    # create text phoneme 
    with open(config.text_phoneme_path, "w", encoding="utf-8") as f:
        metadata.apply(lambda x: create_text_phoneme(f, x["id"], x["score"]), axis=1)
    
    n = 5
    step = int(metadata.shape[0]/n)
    for index in tqdm(range(0, n)):
        # prepare data in kaldi gop format
        print(f"##### Step_1: process data sample from index {index*step} to {(index+1)*step}")
        prepare_data_in_kaldi_gop_format(metadata=metadata[index*step: (index+1)*step], data_dir=config.data_dir)
        print(f"##### Step_1: done !!!")

        print(f"##### Step_2: run kaldi gop")
        os.system("bash run_kaldi_gop.sh")
        print(f"##### Step_2: done !!!")

        print(f"##### Step_3: extract kaldi gop feature")
        extract_kaldi_gop_feature(config=config, metadata=metadata[index*step: (index+1)*step])
        print(f"##### Step_3: done !!!")

        print(f"##### Step_4: generate data for score model")
        feat_path = f'{config.out_dir}/tr_feat_{index}.npy'
        label_path = f'{config.out_dir}/tr_label_{index}.npy'
        generate_sequence_data_for_score_model(config=config, feat_path=feat_path, label_path=label_path)
        print(f"##### Step_4: done !!!")
