import pandas as pd
import json
import numpy as np
import os

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer

from match_label import make_label
from word2vec import W2V


class AudioTextDataset(Dataset):
    def __init__(self, audio_dir, text_dir,
                 sr=16000, n_fft=1024, hop_size=512, n_mels=80,
                 text_max_len=512, audio_max_len=500):
        self.audio_dir = audio_dir
        self.text_dir = text_dir

        self.sr = sr
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.n_mels = n_mels

        self.max_len = text_max_len
        self.audio_max = audio_max_len

        self.mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, n_fft=self.n_fft,
                                                                  hop_length=self.hop_size, n_mels=self.n_mels)

        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")

        self.text_emotion_map = {
            'happy': 'happy',
            'flustered': 'flustered',
            'neutral': 'neutral',
            'angry': 'angry',
            'anxious': 'flustered',
            'hurt': 'sad',
            'sad': 'sad'
        }

        self.neg_emotion_map = {
            'happy': ['flustered', 'neutral', 'angry', 'sad'],
            'flustered': ['happy', 'neutral'],
            'neutral': ['happy', 'flustered', 'angry', 'sad'],
            'angry': ['happy', 'neutral', 'sad'],
            'sad': ['happy', 'angry']
        }

        # text_file_list = ["train_filtered_story_eng_label.json","valid_filtered_story_eng_label.json",
        #                   "test_filtered_story_eng_label.json"]
        # audio_file_list = ['autotagging_moodtheme-train.tsv','autotagging_moodtheme-validation.tsv','autotagging_moodtheme-test.tsv']

        text_file_list = ["valid_filtered_story_eng_label.json"]
        audio_file_list = ['autotagging_moodtheme-validation.tsv']
        self.text_data = pd.concat([self.read_text_data(self.text_dir + file) for file in text_file_list])
        self.audio_data = pd.concat([self.read_jamendo(self.audio_dir + file) for file in audio_file_list])

        self.get_word_vector = W2V().get_vector

        self.text_emotion_idxs, self.audio_emotion_idxs = self.get_emotion_idxes()

    def __len__(self):
        return len(self.audio_data)

    def read_text_data(self, text_path):  # data = json object
        data = json.load(open(text_path))
        ids = [el['recite_src']['id'] for el in data]
        texts = [el['recite_src']['text'] for el in data]
        emotions = [self.text_emotion_map[el['recite_src']['styles'][0]['emotion']] for el in data]
        df = pd.DataFrame(dict(id=ids, text=texts, emotion=emotions))
        return df

    def load_audio_to_mel(self, audio_path):
        audio_sample, sr = torchaudio.load(os.path.join(self.audio_dir, audio_path))
        if sr != self.sr:
            audio_sample = torchaudio.functional.resample(audio_sample, orig_freq=sr, new_freq=self.sr)

        mono = (audio_sample[0] + audio_sample[1]) / 2
        if len(mono) > self.sr * self.audio_max:
            mono = mono[:self.sr * self.audio_max]
        elif len(mono) < self.sr * self.audio_max:
            mono = torch.concat([mono, torch.zeros(self.sr * self.audio_max - len(mono))])
        mel = self.mel_converter(mono)
        mel_file = os.path.join(self.audio_dir, audio_path.split('.')[0] + ("_multi.pt"))
        torch.save(mel,mel_file)
        return mel

    def tokenize_text(self, text):
        text = text.replace("\n", ' ')
        tokenized = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt')
        for key, value in tokenized.items():
            tokenized[key] = torch.squeeze(value)

        return tokenized

    def load_mel(self, mel_path):
        return torch.load(os.path.join(self.audio_dir, mel_path))

    def __getitem__(self, idx):
        audio = self.audio_data.iloc[idx]
        if os.path.exists(os.path.join(self.audio_dir, audio['path'].split('.')[0] + ("_multi.pt"))):
            mel = self.load_mel(audio['path'].split('.')[0] + ("_multi.pt"))
        else :
            mel = self.load_audio_to_mel(audio['path'])

        return {
            'mel': mel ,
            'mel_label': self.get_word_vector(audio['tag']),
            'text': self.get_random_text(audio['text_tag']),            # randomly chosen text sample (positive)
            'text_label': self.get_word_vector(audio['text_tag']),
            'neg_mel': self.get_neg(audio['text_tag'], modal='audio'),  # randomly chosen negative sample
            'neg_text': self.get_neg(audio['text_tag'], modal='text')   # randomly chosen negative text sample
        }

    def read_jamendo(self, audio_path):
        data = open(audio_path).readlines()
        track_id = []
        paths = []
        tags = []
        text_tags = []
        if 'train' in audio_path:
            split = 'train'
        elif 'test' in audio_path:
            split = 'test'
        else:
            split = 'valid'
        for i in range(1, len(data)):
            line = data[i][:-1].split("\t")

            path = line[3]
            tag = line[5:]
            tag = [t.split('---')[-1] for t in tag]

            text_tag = []
            for t in tag:
                tt = self.get_text_tag_from_audio(t)
                if tt in text_tag:
                    continue
                text_tag.append(tt)

            if len(text_tag) == 1:
                track_id.append(line[0])
                paths.append(path)
                tags.append(tag)
                text_tags.append(text_tag)
            else:
                if 'flustered' in text_tag:
                    track_id.append(line[0])
                    paths.append(path)
                    tags.append(tag)
                    text_tags.append(["flustered"])

        df = pd.DataFrame(
            dict(track_id=track_id, path=paths, tag=[x[0] for x in tags], text_tag=[x[0] for x in text_tags]))
        return df.sort_values(by=['text_tag'])

    def get_text_tag_from_audio(self, tag):
        label = make_label()
        if tag in label['happy']['positive']:
            return 'happy'
        if tag in label['flustered']['positive']:
            return 'flustered'
        if tag in label['neutral']['positive']:
            return 'neutral'
        if tag in label['angry']['positive']:
            return 'angry'
        else:
            return 'sad'

    def get_emotion_idxes(self):
        text_idxes = {
            'happy': [],
            'flustered': [],
            'neutral': [],
            'angry': [],
            'sad': []
        }
        audio_idxes = {
            'happy': [],
            'flustered': [],
            'neutral': [],
            'angry': [],
            'sad': []
        }

        for i in range(len(self.text_data)):
            emotion = self.text_data.iloc[i]['emotion']
            text_idxes[emotion].append(self.text_data.iloc[i]['id'])

        for i in range(len(self.audio_data)):
            emotion = self.audio_data.iloc[i]['text_tag']
            audio_idxes[emotion].append(self.audio_data.iloc[i]['track_id'])

        return text_idxes, audio_idxes

    def get_random_text(self, text_tag):  # returns id of randomly chosen text from certain label
        candidates = self.text_emotion_idxs[text_tag]  # emotion 별로 모아놓은 text id
        selected = candidates[np.random.randint(low=0, high=len(candidates) - 1)]
        selected = self.text_data[self.text_data['id'] == selected]['text']
        selected = list(selected)[0]
        return self.tokenize_text(selected)

    def get_neg(self, emotion, modal='text'):
        neg_list = self.neg_emotion_map[emotion]
        # randomly select negative emotion label
        idx = np.random.randint(low=0, high=len(neg_list))
        neg_emotion = neg_list[idx]
        # randomly select text or audio among chosen emotion label
        if modal == 'text':
            candidates = self.text_emotion_idxs[neg_emotion]
        else:
            candidates = self.audio_emotion_idxs[neg_emotion]

        selected = candidates[np.random.randint(low=0, high=len(candidates))]

        if modal == 'text':
            selected = self.text_data[self.text_data['id'] == selected]['text']
            selected = list(selected)[0]
            selected = self.tokenize_text(selected)
        else:
            selected = self.audio_data[self.audio_data['track_id'] == selected]['path']
            selected = list(selected)[0]
            if os.path.exists(os.path.join(self.audio_dir, selected.split('.')[0] + ("_multi.pt"))):
                selected = self.load_mel(selected.split('.')[0] + ("_multi.pt"))
            else:
                selected = self.load_audio_to_mel(selected)

        return selected


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM']='true'
    audio_dir = 'dataset/mtg-jamendo-dataset/'
    text_dir = 'dataset/Story_dataset/'

    dataset = AudioTextDataset(audio_dir=audio_dir, text_dir=text_dir)
    batch = dataset[0]
    batch2 = dataset[1]

    data_loader = DataLoader(dataset, num_workers=4, batch_size=64)

    batch = next(iter(data_loader))
    print(batch['mel_label'].shape)
    print(batch['text_label'].shape)
    print(batch['mel'].shape)
    print(batch['text'])
