# -*- coding: utf-8 -*-
# @Time    : 9/19/21 11:13 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : gen_seq_data_phn.py

# Generate sequence phone input and label for seq2seq models from raw Kaldi GOP features.

import numpy as np
import json
import os
from argparse import ArgumentParser

def load_feat(path):
    file = np.loadtxt(path, delimiter=',')
    return file

def load_keys(path):
    file = np.loadtxt(path, delimiter=',', dtype=str)
    return file

def load_label(path):
    file = np.loadtxt(path, delimiter=',', dtype=str)
    return file

def process_label(label):
    pure_label = []
    for i in range(0, label.shape[0]):
        pure_label.append(float(label[i, 1]))
    return np.array(pure_label)

def process_feat_seq(feat, keys, labels, phn_dict, max_length=50):
    key_set = []
    for i in range(keys.shape[0]):
        cur_key = keys[i].split('.')[0]
        key_set.append(cur_key)

    feat_dim = feat.shape[1] - 1

    utt_cnt = len(list(set(key_set)))
    print('In total utterance number : ' + str(utt_cnt))

    # Pad all sequence to 50 because the longest sequence of the so762 dataset is shorter than 50.
    seq_feat = np.zeros([utt_cnt, max_length, feat_dim])
    # -1 means n/a, padded token
    # [utt, seq_len, 0] is the phone label, and the [utt, seq_len, 1] is the score label
    seq_label = np.zeros([utt_cnt, max_length, 5]) - 1

    # the key format is utt_id.phn_id
    prev_utt_id = keys[0].split('.')[0]

    row = 0
    for i in range(feat.shape[0]):
        cur_utt_id, cur_tok_id = keys[i].split('.')[0], int(keys[i].split('.')[1])
        # if a new sequence, start a new row of the feature vector.
        if cur_utt_id != prev_utt_id:
            row += 1
            prev_utt_id = cur_utt_id

        # The first element is the phone label.
        seq_feat[row, cur_tok_id, :] = feat[i, 1:]

        # phone id
        seq_label[row, cur_tok_id, 0] = phn_dict[labels[i, 0]]
        # phone score
        seq_label[row, cur_tok_id, 1] = labels[i, 1]
        # word score
        seq_label[row, cur_tok_id, 2] = labels[i, 2]
        # word id
        seq_label[row, cur_tok_id, 3] = labels[i, 3]
        # utterance score
        seq_label[row, cur_tok_id, 4] = labels[i, 4]

    return seq_feat, seq_label

def gen_phn_dict(label):
    phn_dict = {}
    phn_idx = 0
    for i in range(label.shape[0]):
        if label[i, 0] not in phn_dict:
            phn_dict[label[i, 0]] = phn_idx
            phn_idx += 1
    return phn_dict

def generate_sequence_data_for_score_model(config, feat_path, label_path):
    tr_feat = load_feat(config.tmp_feat_path)
    tr_keys = load_keys(config.tmp_key_path)
    tr_label = load_label(config.tmp_label_path)

    assert os.path.exists(config.phoneme_dict_path)

    if not os.path.exists(config.phoneme_dict_path):
        phn_dict = gen_phn_dict(tr_label)
        with open(config.phoneme_dict_path, "w", encoding="utf-8") as f:
            json_obj = json.dumps(phn_dict, indent=4, ensure_ascii=False)
            f.write(json_obj)
        print(config.phoneme_dict_path)
    else:
        phn_dict = json.load(open(config.phoneme_dict_path, "r"))

    tr_feat, tr_label = process_feat_seq(tr_feat, tr_keys, tr_label, phn_dict, max_length=config.max_length)

    np.save(feat_path, tr_feat)
    print(f'saved feature to {feat_path}')
    np.save(label_path, tr_label)
    print(f'saved label to {label_path}')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--max_length", default=50, type=int)

    args = parser.parse_args()

    print("Config: ", args)

    # generate sequence training data
    tr_feat = load_feat('../../data/raw_kaldi_gop/gopt_feats/tr_feats.csv')
    tr_keys = load_keys('../../data/raw_kaldi_gop/gopt_feats/tr_keys.csv')
    tr_label = load_label('../../data/raw_kaldi_gop/gopt_feats/tr_labels.csv')
    phn_dict = gen_phn_dict(tr_label)
    # print("Phoneme Dict: ", phn_dict)

    if not os.path.exists("../../resources/phoneme_dict.json"):
        with open("../../resources/phoneme_dict.json", "w", encoding="utf-8") as f:
            json_obj = json.dumps(phn_dict, indent=4, ensure_ascii=False)
            f.write(json_obj)
        print("saved: resources/phoneme_dict.json")

    tr_feat, tr_label = process_feat_seq(tr_feat, tr_keys, tr_label, phn_dict, max_length=args.max_length)
    # print("Train feature shape: ", tr_feat.shape)
    # print("Train label shape: ", tr_label.shape)

    np.save('../../data/seq_data_librispeech/tr_feat.npy', tr_feat)
    # print("Saved train feature to data/seq_data_librispeech/tr_feat.npy")
    np.save('../../data/seq_data_librispeech/tr_label.npy', tr_label)
    # print("Saved train label to data/seq_data_librispeech/tr_label.npy")

    # generate sequence test data
    te_feat = load_feat('../../data/raw_kaldi_gop/gopt_feats/te_feats.csv')
    te_keys = load_keys('../../data/raw_kaldi_gop/gopt_feats/te_keys.csv')
    te_label = load_label('../../data/raw_kaldi_gop/gopt_feats/te_labels.csv')
    te_feat, te_label = process_feat_seq(te_feat, te_keys, te_label, phn_dict, max_length=args.max_length)

    # print("Test feature shape: ", te_feat.shape)
    # print("Test label shape: ", te_label.shape)

    np.save('../../data/seq_data_librispeech/te_feat.npy', te_feat)
    # print("Saved test feature to data/seq_data_librispeech/te_feat.npy")
    np.save('../../data/seq_data_librispeech/te_label.npy', te_label)
    # print("Saved test label to data/seq_data_librispeech/te_label.npy")