

import os
import pandas as pd
import csv
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset

from scripts.label_related import make_label
from dataset.w2v import W2V

from tqdm import tqdm

class MTATDataset:
    def __init__(self, dir_path, split='train', num_max_data=4000, sr=16000):
        self.dir = dir_path
        self.labels = pd.read_csv(self.dir + "meta.csv", index_col=[0])
        self.sr = sr

        if split == "train":
            sub_dir_ids = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c']
        elif split == 'valid':
            sub_dir_ids = ['d']
        else:
            sub_dir_ids = ['e', 'f', 'g']

        is_in_set = [True if x[0] in sub_dir_ids else False for x in self.labels['mp3_path'].values.astype('str')]
        self.labels = self.labels.iloc[is_in_set]
        self.labels = self.labels[:num_max_data]
        self.vocab = self.labels.columns.values[1:-1]
        self.label_tensor = self.convert_label_to_tensor()

    def convert_label_to_tensor(self):
        return torch.LongTensor(self.labels.values[:, 1:-1].astype('bool'))

    def __len__(self):
        return len(self.labels)


class OnMemoryDataset(MTATDataset):
    def __init__(self, dir_path, split='train', num_max_data=8000, sr=16000):
        super().__init__(dir_path, split, num_max_data, sr)

        self.loaded_audios = self.load_audio()

    def load_audio(self):
        total_audio_datas = []

        for audio_path in self.labels['mp3_path']:
            audio_sample, sr = torchaudio.load(self.dir + audio_path)
            if sr != self.sr:
                audio_sample = torchaudio.functional.resample(audio_sample, orig_freq=sr, new_freq=self.sr)
            total_audio_datas.append(audio_sample)

        return total_audio_datas

    def __getitem__(self, idx):
        total_audio_datas = self.load_audio()

        audio_sample = total_audio_datas[idx][0]
        label = self.label_tensor[idx]

        return audio_sample, label


class JamendoDataset(Dataset):
    def __init__(self, dir_path, split='train', sr=16000, audio_max=500, save_data=False):
        self.dir = dir_path
        self.sr = sr
        self.audio_max = audio_max

        if split == "train":
            self.labels = self.read_jamendo(os.path.join(self.dir,'autotagging_moodtheme-train.tsv'))
        elif split == 'valid':
            self.labels = self.read_jamendo(os.path.join(self.dir,'autotagging_moodtheme-validation.tsv'))
        elif split == "test":
            self.labels = self.read_jamendo(os.path.join(self.dir,'autotagging_moodtheme-test.tsv'))
        else:
            train = self.read_jamendo(os.path.join(self.dir, "autotagging_moodtheme-train.tsv"))
            valid = self.read_jamendo(os.path.join(self.dir, "autotagging_moodtheme-validation.tsv"))
            test = self.read_jamendo(os.path.join(self.dir, "autotagging_moodtheme-test.tsv"))
            self.labels = pd.concat([train, valid, test])

        if save_data:
            self.pre_preprocess_and_save_data()

    def __len__(self):
        return len(self.labels)

    def read_jamendo(self, path):
        data = open(path).readlines()
        track_id =[]
        paths=[]
        tags = []
        if 'train' in path:
            split = 'train'
        elif 'test' in path:
            split = 'test'
        else:
            split = 'valid'
        for i in range(1, len(data)):
            line = data[i][:-1].split("\t")

            tag = line[5:]
            tag = [t.split('---')[-1] for t in tag]

            track_id.append(line[0])
            paths.append(os.path.join(split, line[3].replace("/", "-")))
            tags.append(tag)

        df = pd.DataFrame({'TRACK_ID' : track_id,
                           'PATH' : paths,
                           'TAGS' : tags})
        return df

    def pre_preprocess_and_save_data(self):

        print("Saving & Load audio sample")

        for i in range(len(self.labels['PATH'])):
            data = defaultdict(list)
            x = self.labels['PATH'].values[i]

            if os.path.exists(os.path.join(self.dir ,x.split('.')[0] + '.pt')) == True:
                pass
            else:
                audio_sample, sr = torchaudio.load(os.path.join(self.dir, x))

                if sr != self.sr:
                    audio_sample = torchaudio.functional.resample(audio_sample, orig_freq=sr, new_freq=self.sr)
                audio_sample = audio_sample.mean(dim=0)

                if len(audio_sample) > self.sr * self.audio_max:
                    audio_sample = audio_sample[:self.sr * self.audio_max]
                elif len(audio_sample) < self.sr * self.audio_max:
                    audio_sample = torch.concat([audio_sample, torch.zeros(self.sr * self.audio_max - len(audio_sample))])

                data['audio'].append(audio_sample)
                data['label'].append(self.labels['TAGS'][i])

                torch.save(audio_sample, os.path.join(self.dir , x.split('.')[0] + '.pt'))

        print("Saving & Load complete!")

    def load_audio(self):

        total_audio_datas = []

        for audio_path in self.labels['PATH']:
            audio_sample, sr = torchaudio.load(os.path.join(self.dir , audio_path), num_frames=self.sr * self.audio_max)

            if sr != self.sr:
                audio_sample = torchaudio.functional.resample(audio_sample, orig_freq=sr, new_freq=self.sr)
            audio_sample = audio_sample.mean(dim=0)

            total_audio_datas.append(audio_sample)

        return total_audio_datas

    def __getitem__(self, idx):
        '''
        total_audio_datas = self.load_audio()
        audio_sample = total_audio_datas[idx]
        label = self.labels['TAGS'][idx]
        '''

        audio_path = os.path.join(self.dir ,self.labels['PATH'].iloc[idx].split('.')[0] + '.pt')
        audio_sample= torch.load(audio_path)
        # audio_sample = audio_sample[0]

        if audio_sample.size() != 1:
            audio_sample = audio_sample.squeeze(0)

        return {"audio" : audio_sample,
                "label" : label[0]
        }


class MelSpectrogram(nn.Module):
    def __init__(self,
                 sr=16000,
                 n_fft = 1024,
                 hop_length = 512,
                 n_mels = 80):
        super().__init__()
        self.mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  n_fft=n_fft,
                                                                  hop_length=hop_length,
                                                                  n_mels=n_mels)
    def forward(self, input):
        mel_spec = self.mel_converter(input)

        return mel_spec


class AudioOnlyDataset(Dataset):
    def __init__(self, audio_dir, split='train', device='cpu',
                 sr=16000, n_fft=1024, hop_size=512, n_mels=80,
                 max_len=500):

        self.audio_dir = audio_dir
        self.device = device

        self.sr = sr
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.audio_max = max_len

        self.mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, n_fft=self.n_fft,
                                                                  hop_length=self.hop_size, n_mels=self.n_mels)

        self.neg_emotion_map = {
            'happy': ['flustered', 'neutral', 'angry', 'sad'],
            'flustered': ['happy', 'neutral'],
            'neutral': ['happy', 'flustered', 'angry', 'sad'],
            'angry': ['happy', 'neutral', 'sad'],
            'sad': ['happy', 'angry']
        }

        audio_file_list = ['autotagging_moodtheme-train.tsv', 'autotagging_moodtheme-validation.tsv',
                           'autotagging_moodtheme-test.tsv']


        if split=='train':
            self.audio_data = self.read_jamendo(self.audio_dir + audio_file_list[0])
        elif split=='valid':
            self.audio_data = self.read_jamendo(self.audio_dir + audio_file_list[1])
        else:
            self.audio_data = self.read_jamendo(self.audio_dir + audio_file_list[2])

        self.loaded_audio = self.load_all_audio()

        self.get_word_vector = W2V().get_vector

        self.audio_emotion_idxs = self.get_emotion_idxes()

    def __len__(self):
        return len(self.audio_data)

    def load_all_audio(self):
        audios={}
        print("Load all audios")
        for path in tqdm(self.audio_data['path']):
            if os.path.exists(os.path.join(self.audio_dir, path.replace("mp3", "pt"))):
                mel = self.load_mel(path.replace("mp3", "pt"))
            else:
                mel = self.load_audio_to_mel(path)
            audios[path]=mel
        return audios

    def load_audio_to_mel(self, audio_path):
        audio_sample, sr = torchaudio.load(os.path.join(self.audio_dir, audio_path))
        if sr != self.sr:
            audio_sample = torchaudio.functional.resample(audio_sample, orig_freq=sr, new_freq=self.sr)

        if len(audio_sample) > 1:
            audio_sample = audio_sample.mean(dim=0)

        if len(list(audio_sample.shape)) > 1:
            audio_sample = audio_sample.squeeze()

        if len(audio_sample) > self.sr * self.audio_max:
            audio_sample = audio_sample[:self.sr * self.audio_max]
        elif len(audio_sample) < self.sr * self.audio_max:
            audio_sample = torch.concat([audio_sample, torch.zeros(self.sr * self.audio_max - len(audio_sample))])
        mel = self.mel_converter(audio_sample)
        mel_file = os.path.join(self.audio_dir, audio_path.replace("mp3","pt"))
        torch.save(mel,mel_file)
        return mel

    def load_mel(self, mel_path):
        return torch.load(os.path.join(self.audio_dir, mel_path))

    def __getitem__(self, idx):
        audio = self.audio_data.iloc[idx]
        mel = self.loaded_audio[audio['path']]

        return {
            'mel': mel.to(self.device),
            'mel_label': self.get_word_vector(audio['tag']).to(self.device),
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

            path = os.path.join(split, line[3].replace("/", "-"))

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
        audio_idxes = {
            'happy': [],
            'flustered': [],
            'neutral': [],
            'angry': [],
            'sad': []
        }


        for i in range(len(self.audio_data)):
            emotion = self.audio_data.iloc[i]['text_tag']
            audio_idxes[emotion].append(self.audio_data.iloc[i]['track_id'])

        return audio_idxes
