from flask import Flask, request, jsonify
import torch.nn.functional as F
import soundfile as sf
from model import GOPT
import numpy as np
import pandas as pd
import os
import torch
import pickle
import json
import io

app = Flask(__name__)

device = "cuda:0"
def load_scaler(path):
    with open(path, "rb") as f:
        scaler = pickle.load(f)

    return scaler

def init_model(path="exp/models/best_audio_model.pth"):
    gopt = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=84)
    sd = torch.load(path, map_location='cpu')
    gopt.load_state_dict(sd)
    gopt.eval()
    gopt = gopt.to(device)

    return gopt

def load_lexicon(path="resources/lexicon.txt"):
    with open(path, 'r') as f:
        lexicon_raw = f.read()
        rows = lexicon_raw.splitlines()
    clean_rows = [row.split() for row in rows]
    lexicon_dict = dict()
    for row in clean_rows:
        c_row = row.copy()
        key = c_row.pop(0)
        if len(c_row) == 1:
            c_row[0] = c_row[0] + '_S'
        if len(c_row) >= 2:
            c_row[0] = c_row[0] + '_B'
            c_row[-1] = c_row[-1] + '_E'
        if len(c_row) > 2:
            for i in range(1,len(c_row)-1):
                c_row[i] = c_row[i] + '_I'
        val = " ".join(c_row)
        lexicon_dict[key] = val
    return lexicon_dict

def parse_score(transcript, utt_score, word_scores, word_ids, phn_scores, phone_ids):
    print("TEXT: ", transcript)
    print("PHONEME IDS: ", phone_ids)
    print("WORD IDS: ", word_ids)
    words = transcript.split(" ")
    print(words)
    curr_word_id = -1
    _tmp_word = {
            "text": words[0],
            "score": word_scores[0].item(),
            "phonemes": []
        }

    scores = {
        "version": "None",
        "utterance": {
            "text": transcript,
            "score": utt_score.item(),
            "words": []
        }
    }
    for index in range(len(phone_ids)):
        word_id = word_ids[index]

        _tmp_phone = {
            "text": id2phone[int(phone_ids[index])],
            "score": phn_scores[index].item(),
        }

        if word_id == curr_word_id:
            scores["utterance"]["words"][word_id]["phonemes"].append(
                _tmp_phone
            )
        else:
            _tmp_word = {
                "text": words[word_id],
                "score": word_scores[word_id].item(),
                "phonemes": []
            }

            _tmp_word["phonemes"].append(_tmp_phone)
            if len(scores["utterance"]["words"]) == word_id:
                scores["utterance"]["words"].append(_tmp_word)
            else:
                scores["utterance"]["words"][word_id]["phonemes"].append(
                    _tmp_phone
                )
    
    return scores


def load_phone2id_and_id2phone(path="resources/phoneme_dict.json"):
    with open(path, "r") as f:
        phone2id = json.load(f)
    id2phone = {value:key for key, value in phone2id.items()}

    return phone2id, id2phone

@app.route('/')
def index():
    return {
        "status": "success"
    }

@app.route('/pronunciation_scoring', methods=['POST'])
def upload():
    file = request.files.get('wav')
    data, samplerate = sf.read(io.BytesIO(file.read()))
    sf.write("/working/audio.wav", data, samplerate)

    transcript = request.form.get('transcript')
    print("transcript: ", transcript)

    os.system(f'bash run_feature_extraction.sh "/working/audio.wav" "{transcript}"')
    input_feat = np.load("data/seq_data_librispeech/if_feat.npy")
    input_phn = np.load("data/seq_data_librispeech/if_label.npy")

    normed_feat = scaler.transform(input_feat[0])
    input_feat = torch.from_numpy(normed_feat).unsqueeze(0)
    phoneme_length = np.sum(input_phn!=-1)

    with torch.no_grad():
        input_feats = input_feat.to(device).float()
        input_phone_ids = torch.from_numpy(input_phn[:,:,0]).to(device).float()
        utt_score, phn_scores, wrd_scores = gopt(input_feats, input_phone_ids)
    
    phn_scores = phn_scores.view(-1)[0: phoneme_length] * 50
    utt_score = utt_score.view(-1) * 50
    wrd_scores = wrd_scores.view(-1)[0:phoneme_length] * 50

    input_phone_ids = input_phone_ids.view(-1)
    phone_ids = input_phone_ids[input_phone_ids != -1]
    
    text_phone_path = "egs/gop_speechocean762/s5/data/local/text-phone"
    phone_df = pd.read_csv(text_phone_path, sep="\t", names=["word_id", "phonemes"], dtype={"word_id":str})
    
    phone_df.word_id = phone_df.word_id.apply(lambda x: x.split(".")[-1])
    phone_df.phonemes = phone_df.phonemes.apply(lambda x: x.split(" "))
    phone_df = phone_df.explode(column="phonemes").reset_index()
    phone_df.phonemes = phone_df.phonemes.apply(lambda x: x.split("_")[0])

    word_ids = torch.tensor([int(i) for i in phone_df["word_id"].to_list()])
    one_hot = F.one_hot(word_ids, num_classes=word_ids.max().item()+1).float()
    one_hot = one_hot / one_hot.sum(0, keepdim=True)
    word_scores = torch.matmul(one_hot.transpose(0, 1), phn_scores.cpu())

    scores = parse_score(
        transcript=transcript, utt_score=utt_score, word_scores=word_scores,
        word_ids=word_ids, phn_scores=phn_scores, phone_ids=phone_ids)
    
    # with open("result.json", "w", encoding="utf-8") as f:    
    #     json_obj = json.dumps(scores, indent=4, ensure_ascii=False)
    #     f.write(json_obj)

    return scores

if __name__ == "__main__":
    gopt = init_model(path="exp/models/best_audio_model.pth")
    scaler = load_scaler(path="resources/scaler.pkl")
    lexicon_dict = load_lexicon(path="resources/lexicon.txt")

    with open("resources/phoneme_dict.json", "r") as f:
        phone2id = json.load(f)
    id2phone = {value:key for key, value in phone2id.items()}

    app.run(host="0.0.0.0", debug=False, port=6666)