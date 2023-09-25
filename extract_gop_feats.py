import pandas as pd
import kaldi_io
import os
from glob import glob
from tqdm import tqdm
import json
import numpy as np

def preprocess_word(words):
    processed_words = []
    for word in words:
        if len(word[1].split()) == 1:
            processed_words.append(word)
        else:
            for phn in word[1].split(" "):
                processed_words.append([word[0], phn, word[2]])
    return processed_words

def create_metadata_dicts(metadata):
    word2score, phone2score, utt2score = {}, {}, {}
    phone2wordid = {}

    def extract_score(utt_id, score):
        score = json.loads(score)
        words = score["words"]
        phonemes = score["phonemes"]

        assert len(words) == len(phonemes)
        assert utt_id not in utt2score
        utt2score[str(utt_id)] = score["utterance"]
        index = 0
        
        for wrd_id, (word, phoneme) in enumerate(zip(words, phonemes)):
            phoneme = preprocess_word(phoneme)
            assert len(word[1].split()) == len(phoneme)
            for _, (x, y ) in enumerate(zip(word[1].split(), phoneme)):
                key = f'{utt_id}.{index}'
                
                assert key not in word2score
                assert key not in phone2score
                
                word2score[key] = word[-1]
                phone2score[key] = y[-1]
                phone2wordid[key] = wrd_id
                
                index+=1
                
    metadata.apply(lambda x: extract_score(x["id"], x["score"]), axis=1)

    return word2score, phone2score, utt2score, phone2wordid

def load_phone_symbol_table(filename):
    if not os.path.isfile(filename):
        return None, None
    int2sym = {}
    sym2int = {}
    with open(filename, 'r') as f:
        for line in f:
            sym, idx = line.strip('\n').split('\t')
            idx = int(idx)
            int2sym[idx] = sym
            sym2int[sym] = idx
    return sym2int, int2sym

def extract_kaldi_gop_feature(config, metadata):
    word2score, phone2score, utt2score, phone2wordid = create_metadata_dicts(metadata)
    _, phone_int2sym = load_phone_symbol_table(config.phone2sym_path)

    os.chdir(config.gop_path)
    keys, features, labels = [], [], []
    for phn_id, feature in tqdm(kaldi_io.read_vec_flt_scp(config.feat_path)):
        phn_id = str(phn_id)
        uut_id = phn_id.split(".")[0]
        
        features.append(feature)
        keys.append(phn_id)
        
        phoneme = phone_int2sym[feature[0]]
        labels.append(
            [
                phoneme, 
                phone2score[phn_id],
                word2score[phn_id],
                phone2wordid[phn_id],
                utt2score[uut_id],
                ]
        )
    os.chdir(config.root_dir)

    np.savetxt(config.tmp_feat_path, features, delimiter=',')
    np.savetxt(config.tmp_key_path, keys, delimiter=',', fmt='%s')
    np.savetxt(config.tmp_label_path, labels, delimiter=',', fmt='%s')
