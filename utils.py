from glob import glob 
from tqdm import tqdm
import pandas as pd
import numpy as np
import kaldi_io
import shutil
import uuid
import json
import time
import os

def create_wav_scp(f, wav_id, wav_path):
    line = f'{wav_id}\t{wav_path}'
    
    f.write(line + "\n")
    
def create_text(f, utt_id, text):
    line = f'{utt_id}\t{text.upper()}'
    
    f.write(line+"\n")
    
def create_utt2spk(f, utt_id, spk):
    line = f'{utt_id}\t{spk}'
    
    f.write(line+"\n")
    
def create_spk2utt(f, spk, utt_id):
    line = f'{spk}\t{utt_id}'
    
    f.write(line+"\n")
    
def gen_spk_id(*args):
    return uuid.uuid1()

def create_text_phoneme(f, utt_id, words):
    words = json.loads(words)["words"]
    for index, word in enumerate(words):
        phonemes = word[1].split()
        if len(phonemes) == 1:
            phonemes[0] = phonemes[0] + "_S"
            
        if len(phonemes) >= 2:
            phonemes[0] = phonemes[0] + "_B"
            phonemes[-1] = phonemes[-1] + "_E"
            
        if len(phonemes) > 2:
            for i in range(1, len(phonemes)-1):
                phonemes[i] = phonemes[i] + "_I"
            
        line = f'{utt_id}.{index}\t{" ".join(phonemes)}'
        f.write(line + "\n")

def check_downloaded(path):
    if os.path.exists(path):
        return False
    return True

def prepare_data_in_kaldi_gop_format(metadata, data_dir):
    wavscp_path = f'{data_dir}/wav.scp'
    text_path = f'{data_dir}/text'
    spk2utt_path = f'{data_dir}/spk2utt'
    utt2spk_path = f'{data_dir}/utt2spk'

    with open(wavscp_path, "w", encoding="utf-8") as f:
        metadata.sort_values("id").apply(lambda x: create_wav_scp(f, x["id"], x["wav_path"]), axis=1)
        
    with open(text_path, "w", encoding="utf-8") as f:
        metadata.apply(lambda x: create_text(f, x["id"], x["text"]), axis=1)
        
    with open(spk2utt_path, "w", encoding="utf-8") as f:
        metadata.sort_values("spk_id").apply(lambda x: create_spk2utt(f, x["spk_id"], x["id"]), axis=1)
        
    with open(utt2spk_path, "w", encoding="utf-8") as f:
        metadata.sort_values("id").apply(lambda x: create_utt2spk(f, x["id"], x["spk_id"]), axis=1)

def extract_score(score):
    score = json.loads(score)
    word_scores = []
    for word in score["words"]:
        word_scores.append(word[2])
    
    return word_scores
