# Copyright 2021  Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

# This script trains models to convert GOP-based feature into human
# expert scores.

# 1c is as 1b, but use SVR instead of random forest regression.
# Comparing with 1b, the f1-score of the class 0 is much improved.

# MSE: 0.16
# Corr: 0.45
#
#               precision    recall  f1-score   support
#
#            0       0.42      0.30      0.35      1339
#            1       0.16      0.36      0.22      1828
#            2       0.97      0.92      0.94     44079
#
#     accuracy                           0.88     47246
#    macro avg       0.52      0.53      0.50     47246
# weighted avg       0.92      0.88      0.90     47246


import sys
import argparse
import pickle
import kaldi_io
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.svm import SVR
import os
import re
import json
import random
from itertools import chain
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
# from utils import (load_phone_symbol_table,
#                    load_human_scores,
#                    add_more_negative_data)
def round_score(score, floor=0.1, min_val=0, max_val=2):
    score = max(min(max_val, score), min_val)
    return round(score / floor) * floor

def load_human_scores(filename, floor=0.1):
    with open(filename) as f:
        info = json.load(f)
    score_of = {}
    phone_of = {}
    for utt in info:
        phone_num = 0
        for word in info[utt]['words']:
            assert len(word['phones']) == len(word['phones-accuracy'])
            for i, phone in enumerate(word['phones']):
                key = f'{utt}.{phone_num}'
                phone_num += 1
                phone_of[key] = re.sub(r'[_\d].*', '', phone)
                score_of[key] = round_score(word['phones-accuracy'][i], floor)
    return score_of, phone_of

def add_more_negative_data(data):
    # Put all examples together
    whole_data = []
    for ph in data:
        for examples in data[ph]:
            whole_data.append(list(chain(*([ph], examples))))

    # Take the 2-score examples of other phones as the negative examples
    for cur_ph in data:
        labels, feats = list(zip(*data[cur_ph]))
        count_of_label = Counter(labels)
        example_number_needed = 2 * count_of_label[2] - len(labels)
        if example_number_needed > 0:
            features = random.sample([feat for ph, score, feat in whole_data
                                      if ph != cur_ph and score == 2],
                                     example_number_needed)
            examples = list(zip([0] * example_number_needed, features))
            data[cur_ph] = data[cur_ph] + examples
    return data


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

def get_args():
    parser = argparse.ArgumentParser(
        description='Train a simple polynomial regression model to convert '
                    'gop into human expert score',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--phone-symbol-table', type=str, default='',
                        help='Phone symbol table, used for detect unmatch '
                             'feature and labels')
    parser.add_argument('--nj', type=int, default=1, help='Job number')
    parser.add_argument('feature_scp',
                        help='Input gop-based feature file, in Kaldi scp')
    parser.add_argument('human_scoring_json',
                        help='Input human scores file, in JSON format')
    parser.add_argument('model', help='Output the model file')
    sys.stderr.write(' '.join(sys.argv) + "\n")
    args = parser.parse_args()
    return args


def train_model_for_phone(label_feat_pairs):
    model = SVR()
    labels, feats = list(zip(*label_feat_pairs))
    labels = np.array(labels).reshape(-1, 1)
    feats = np.array(feats).reshape(-1, len(feats[0]))
    labels = labels.ravel()
    model.fit(feats, labels)
    return model


def main():
    args = get_args()

    # Phone symbol table
    _, phone_int2sym = load_phone_symbol_table(args.phone_symbol_table)

    # Human expert scores
    score_of, phone_of = load_human_scores(args.human_scoring_json, floor=1)

    # Prepare training data
    train_data_of = {}
    for ph_key, feat in kaldi_io.read_vec_flt_scp(args.feature_scp):
        if ph_key not in score_of:
            print(f'Warning: no human score for {ph_key}')
            continue
        ph = int(feat[0])
        if phone_int2sym is not None:
            if phone_int2sym[ph] != phone_of[ph_key]:
                print(f'Unmatch: {phone_int2sym[ph]} <--> {phone_of[ph_key]} ')
                continue
        score = score_of[ph_key]
        train_data_of.setdefault(ph, []).append((score, feat[1:]))

    # Make the dataset more blance
    train_data_of = add_more_negative_data(train_data_of)

    # Train models
    with ProcessPoolExecutor(args.nj) as ex:
        future_to_model = [(ph, ex.submit(train_model_for_phone, pairs))
                           for ph, pairs in train_data_of.items()]
        model_of = {ph: future.result() for ph, future in future_to_model}

    # Write to file
    with open(args.model, 'wb') as f:
        pickle.dump(model_of, f)


if __name__ == "__main__":
    main()
