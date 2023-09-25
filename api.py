from flask import Flask, request, jsonify
from dataclasses import dataclass
from typing import List
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

device = "cuda:0"
app = Flask(__name__)

@dataclass
class Phoneme:
    arpabet: str
    start_time: float
    end_time: float
    score: float
    
    def __init__(self, arpabet, start_time, end_time, score):
        self.arpabet = arpabet
        self.start_time = start_time
        self.end_time = end_time
        self.score = score
    
    def to_dict(self):
        return {
            "arpabet": self.arpabet,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "score": self.score
        }
        
@dataclass
class Word:
    text: str
    arpabet: str
    start_time: float
    phonemes: List[Phoneme]
    end_time: float
    score: float
    
    def __init__(self, text, arpabet, start_time, end_time, score, phonemes):
        self.arpabet = arpabet
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        self.score = score
        self.phonemes = phonemes
    
    def append_phone(self, phone):
        self.phonemes.append(phone)
    
    def to_dict(self):
        return {
            "text": self.text,
            "arpabet": self.arpabet,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "score": self.score,
            "phonemes": [phoneme.to_dict() for phoneme in self.phonemes]

        }

@dataclass
class Sentence:
    arpabet: str
    duration: float
    score: float
    words: List[Word]
    
    def __init__(self, text, arpabet, duration, score, words):
        self.arpabet = arpabet
        self.duration = duration
        self.text = text
        self.score = score
        self.words = words
        
    def append_word(self, word):
        self.words.append(word)
    
    def append_phoneme(self, word_index, phoneme):
        self.words[word_index].append_phone(phoneme)
        self.words[word_index].end_time = phoneme.end_time 
    
    def to_dict(self):
        return {
            "text": self.text,
            "arpabet": self.arpabet,
            "score": self.score,
            "duration": self.duration,
            "phonemes": [word.to_dict() for word in self.words]
        }

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

def parse_score(metadata, transcript, utt_score, word_scores, word_ids, phn_scores, phone_ids):
    words = transcript.split(" ")

    _tmp_word = Word(
        text=words[0], 
        arpabet=None,
        score=word_scores[0].item(), 
        end_time=0, 
        start_time=0, 
        phonemes=[]
    )
    
    utterance = Sentence(
        text=transcript,
        arpabet=None, 
        score=utt_score.item(), 
        duration=0.0, 
        words=[],
    )
    curr_word_id = -1
    for index in range(len(phone_ids)):
        word_id = int(metadata["word_id"][index])
        start_time = metadata["start"][index]
        end_time = start_time + metadata["duration"][index]
        
        _tmp_phone = Phoneme(
            arpabet=id2phone[int(phone_ids[index])],
            end_time=end_time, start_time=start_time,
            score=phn_scores[index].item()
        )

        if word_id == curr_word_id:
            utterance.append_phoneme(word_index=word_id, phoneme=_tmp_phone)
        else:
            _tmp_word = Word(
                arpabet=None, start_time=_tmp_phone.start_time, end_time=_tmp_phone.end_time, text=words[word_id],
                score=word_scores[word_id].item(), phonemes=[_tmp_phone, ]
            )
            if len(utterance.words) == word_id:
                utterance.append_word(_tmp_word)
            else:
                utterance.append_phoneme(word_index=word_id, phoneme=_tmp_phone)
    scores = {
        "version": "None",
        "utterance": utterance.to_dict()
        }

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

def load_force_alignment_result(alignment_path, phones_path):
    alignment = pd.read_csv(alignment_path, sep="\s", names=["file_utt","utt","start","duration","id"], engine='python')
    
    id2phoneme = pd.read_csv(phones_path, sep="\s", names=["phonemes", "id"], engine='python')
    id2phoneme = id2phoneme.set_index(keys="id").to_dict()["phonemes"]
    alignment["phonemes"] = alignment.id.apply(lambda x: id2phoneme[int(x)])
    
    return alignment

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
    
    os.system("bash run_force_alignment.sh")
    
    alignment_path = "egs/gop_speechocean762/s5/exp/ali_infer/merged_alignment.txt"
    phones_path = "egs/gop_speechocean762/s5/data/lang_nosp/phones.txt"

    alignment = load_force_alignment_result(alignment_path=alignment_path, phones_path=phones_path)
    alignment.phonemes = alignment.phonemes.apply(lambda x: x.split(" "))
    alignment = alignment.explode(column="phonemes").reset_index()
    alignment["pure_phonemes"] = alignment.phonemes.apply(lambda x: x.split("_")[0])
    alignment = alignment[alignment.phonemes != "SIL"].reset_index()
    
    text_phone_path = "egs/gop_speechocean762/s5/data/local/text-phone"
    text_phone_df = pd.read_csv(text_phone_path, sep="\t", names=["word_id", "phonemes"], dtype={"word_id":str})
    
    text_phone_df.word_id = text_phone_df.word_id.apply(lambda x: x.split(".")[-1])
    text_phone_df.phonemes = text_phone_df.phonemes.apply(lambda x: x.split(" "))
    text_phone_df = text_phone_df.explode(column="phonemes").reset_index()
    text_phone_df["pure_phonemes"] = text_phone_df.phonemes.apply(lambda x: x.split("_")[0])
    
    joined_metadata = pd.concat([text_phone_df, alignment[["start", "duration", "phonemes", "pure_phonemes"]]], axis=1)
    
    print(joined_metadata)
    word_ids = torch.tensor([int(i) for i in joined_metadata["word_id"].to_list()])
    one_hot = F.one_hot(word_ids, num_classes=word_ids.max().item()+1).float()
    one_hot = one_hot / one_hot.sum(0, keepdim=True)
    word_scores = torch.matmul(one_hot.transpose(0, 1), phn_scores.cpu())

    scores = parse_score(metadata=joined_metadata,
        transcript=transcript, utt_score=utt_score, word_scores=word_scores,
        word_ids=word_ids, phn_scores=phn_scores, phone_ids=phone_ids)
    
    with open("result.json", "w", encoding="utf-8") as f:    
        json_obj = json.dumps(scores, indent=4, ensure_ascii=False)
        f.write(json_obj)

    return scores

if __name__ == "__main__":
    gopt = init_model(path="exp/models/best_audio_model.pth")
    scaler = load_scaler(path="resources/scaler.pkl")
    lexicon_dict = load_lexicon(path="resources/lexicon.txt")

    with open("resources/phoneme_dict.json", "r") as f:
        phone2id = json.load(f)
    id2phone = {value:key for key, value in phone2id.items()}

    app.run(host="0.0.0.0", debug=False, port=6666)