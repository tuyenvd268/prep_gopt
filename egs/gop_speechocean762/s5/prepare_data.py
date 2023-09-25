import pandas as pd
import argparse
import os

def load_lexicon(path="librispeech-lexicon.txt"):
    with open(path, 'r') as f:
        lexicon_raw = f.read()
        rows = lexicon_raw.splitlines()
    clean_rows = [row.split() for row in rows]
    lexicon_dict_l = dict()
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
        lexicon_dict_l[key] = val
    return lexicon_dict_l

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path')
parser.add_argument('-t', '--text')
args = parser.parse_args()

utterance =  args.text
wav_path =  args.path
print("PREPARE DATA FOR KALDI: ", utterance)

user_id = "0000"
# utterance =  "THE PAPER PUBLISHED NO DETAILS AND THERE WAS NO ATTACK"
# wav_path = "/working/egs/gop_speechocean762/s5/data/speechocean762/WAVE/SPEAKER1030/010300105.WAV"
lexicon_path = "/working/resources/lexicon.txt"
lexicon_dict_l = load_lexicon(lexicon_path)

sample = {
    "user_id":[user_id, ],
    "utterance":[utterance, ],
    "wav_path":[wav_path, ]
}
dataset = pd.DataFrame(sample)
dataset.head()

base_path = "/working/egs/gop_speechocean762/s5/data/infer"
scp_file = f'{base_path}/wav.scp'
spk2utt = f'{base_path}/spk2utt'
utt2spk = f'{base_path}/utt2spk'
text = f'{base_path}/text'
text_phone =  f'data/local/text-phone'

scp_str = ""
spk2utt_str = ""
utt2spk_str = ""
text_str = ""
text_phone_str = ""

for index in dataset.index:
    wav_id = os.path.basename(dataset["wav_path"][index]).rstrip(".WAV").rstrip(".wav")
    
    _scp_str = f'{wav_id}\t{dataset["wav_path"][index]}\n'
    _text_str = f'{wav_id}\t{dataset["utterance"][index]}\n'
    _utt2spk_str = f'{wav_id}\t{dataset["user_id"][index]}\n'
    _spk2utt_str = f'{dataset["user_id"][index]}\t{wav_id}\n'
        
    for i, word in enumerate(dataset["utterance"][index].split()):
        _text_phone_str = f'{wav_id}.{i}\t{lexicon_dict_l[word]}\n'
        text_phone_str += _text_phone_str
        
    scp_str += _scp_str
    text_str += _text_str
    utt2spk_str += _utt2spk_str
    spk2utt_str += _spk2utt_str
        

with open(scp_file, 'w') as f:
    f.write(scp_str)

with open(spk2utt, 'w') as f:
    f.write(spk2utt_str)

with open(utt2spk, 'w') as f:
    f.write(utt2spk_str)

with open(text, 'w') as f:
    f.write(text_str)

with open(text_phone, 'w') as f:
    f.write(text_phone_str)