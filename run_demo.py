import torch
import sys
import os
from model import GOPT
import numpy as np
import json
import pandas as pd 
import streamlit as st
import pickle
import requests
import json
import os

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

def load_lexicon(path="librispeech-lexicon.txt"):
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


def load_label(path):
    file = np.loadtxt(path, delimiter=',', dtype=str)
    return file

gopt = init_model()
scaler = load_scaler(path="resources/scaler.pkl")

lexicon_path = "resources/lexicon.txt"
lexicon_dict_l = load_lexicon(lexicon_path)

with open("resources/phoneme_dict.json", "r") as f:
    phone2id = json.load(f)
id2phone = {value:key for key, value in phone2id.items()}

st.set_page_config(page_title="Pronunciation Scoring", page_icon="👄")
st.image(
    "https://emojipedia-us.s3.amazonaws.com/source/skype/289/parrot_1f99c.png",
    width=125,
)

st.title("Pronunciation Scoring")
text_input = st.text_input("Enter some text 👇")

c1, c2, c3 = st.columns([1, 4, 1])

with c2:
    with st.form(key="my_form"):
        f = st.file_uploader("", type=[".wav"])
        submit_button = st.form_submit_button(label="Scoring")

if f is not None:
    st.audio(f, format="wav")
    path_in = f.name

    old_file_position = f.tell()
    f.seek(0, os.SEEK_END)
    getsize = f.tell()
    f.seek(old_file_position, os.SEEK_SET)
    getsize = round((getsize / 1000000), 1)

    if getsize < 5:  # File more than 5 MB
        bytes_data = f.getvalue()

        st.audio(f, format="wav")

        path = f'{os.getcwd()}/audio.WAV'
        with open(path, mode='wb') as f:
            f.write(bytes_data)
            
        # run feature extraction using kaldi pipline
        os.system(f'bash run_feature_extraction.sh "{path}" "{text_input}"')
        st.info(text_input)

        input_feat = np.load("data/seq_data_librispeech/if_feat.npy")
        input_phn = np.load("data/seq_data_librispeech/if_label.npy")

        normed_feat = scaler.transform(input_feat[0])
        input_feat = torch.from_numpy(normed_feat).unsqueeze(0)
        
        with torch.no_grad():
            t_input_feat = input_feat.to(device)
            t_phn = torch.from_numpy(input_phn[:,:,0]).to(device)

            utt_score, phn_score, wrd_score = gopt(t_input_feat.float(),t_phn.float())
            
        # merge phoneme input with score
        phn_id = t_phn.view(-1)
        phone_with_scores = -1 * np.ones((phn_id[phn_id != -1].shape[0], 2))
        phn_score = phn_score.view(-1)

        for i in range(phone_with_scores.shape[0]):
            if phn_id[i] == -1:
                continue 
            phone_with_scores[i, 0] = phn_id[i]
            phone_with_scores[i, 1] = phn_score[i]*50

        print("phone score: ", phone_with_scores)

        phone_with_scores = []
        for sample in input_phn:
            if int(sample[0]) == -1:
                continue
            phone = id2phone[int(sample[0])]
            phone_score =  sample[1]
            
            phone_with_scores.append([phone, phone_score])
            
        print(phone_with_scores)

        words = text_input.split(' ')
        text_phone_path = "egs/gop_speechocean762/s5/data/local/text-phone"
        phone_df = pd.read_csv(text_phone_path, sep="\t", names=["word_id", "phonemes"], dtype={"word_id":str})
        
        phone_df.word_id = phone_df.word_id.apply(lambda x: x.split(".")[-1])
        phone_df.phonemes = phone_df.phonemes.apply(lambda x: x.split(" "))
        phone_df = phone_df.explode(column="phonemes").reset_index()
        phone_df.phonemes = phone_df.phonemes.apply(lambda x: x.split("_")[0])
        
        results = [None,]*len(text_input.split(" "))
        for index in range(len(phone_with_scores)):
            index_int = int(phone_df["word_id"][index])
            if results[index_int] is None:
                results[index_int] = {
                    "text": words[index_int],
                    "phoneme_with_score":[[phone_df["phonemes"][index], phone_with_scores[index][1]]]
                    }
            else:
                # assert phone_with_scores[index][0] == phone_df["phonemes"][index]
                results[index_int]["phoneme_with_score"].append([phone_df["phonemes"][index], phone_with_scores[index][1]])
        with open("result.json", "w", encoding="utf-8") as f:
            json_obj = json.dumps(results, indent=4, ensure_ascii=False)
            f.write(json_obj)

        # st.info(json.dumps(final_score, indent=4, ensure_ascii=False))
        st.json(results)
    else:
        st.warning(
            "🚨 We've limited this demo to 5MB files. Please upload a smaller file."
        )
        st.stop()

else:
    path_in = None
    st.stop()

