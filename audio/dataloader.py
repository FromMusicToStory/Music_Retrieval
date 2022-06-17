import os
import pandas as pd
import csv
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torchaudio
import torch.utils.data as data


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


class JamendoDataset(data.Dataset):
    def __init__(self, dir_path, split='train', sr=16000, audio_max=500):
        self.dir = dir_path
        self.sr = sr
        self.audio_max = audio_max

        if split == "train":
            self.labels = self.read_jamendo(self.dir + 'autotagging_moodtheme-train.tsv')
        elif split == 'valid':
            self.labels = self.read_jamendo(self.dir + 'autotagging_moodtheme-validation.tsv')
        elif split == "test":
            self.labels = self.read_jamendo(self.dir + 'autotagging_moodtheme-test.tsv')
        else:
            train = self.read_jamendo(self.dir + "autotagging_moodtheme-train.tsv")
            valid = self.read_jamendo(self.dir + "autotagging_moodtheme-validation.tsv")
            test = self.read_jamendo(self.dir + "autotagging_moodtheme-test.tsv")
            self.labels = pd.concat([train, valid, test])

        self.pre_preprocess_and_save_data()

    def __len__(self):
        return len(self.labels)

    def read_jamendo(self, path):
        data = open(path).readlines()
        track_id =[]
        paths=[]
        tags = []

        for i in range(1, len(data)):
            line = data[i][:-1].split("\t")

            tag = line[5:]
            tag = [t.split('---')[-1] for t in tag]

            track_id.append(line[0])
            paths.append(line[3])
            tags.append(tag)

        df = pd.DataFrame({'TRACK_ID' : track_id,
                           'PATH' : paths,
                           'TAGS' : tags})
        return df

    def pre_preprocess_and_save_data(self):

        print("audio sample Saving & Load")

        for i in range(len(self.labels['PATH'])):
            data = defaultdict(list)

            x = self.labels['PATH'].values[i]

            if os.path.exists(self.dir + x.split('.')[0] + '.pt') == True:
                pass
            else:
                audio_sample, sr = torchaudio.load(self.dir + x, num_frames=self.sr * self.audio_max)

                if len(audio_sample) > 1:
                    audio_sample = audio_sample.mean(dim=0)

                if sr != self.sr:
                    audio_sample = torchaudio.functional.resample(audio_sample, orig_freq=sr, new_freq=self.sr)

                data['audio'].append(audio_sample)
                data['label'].append(self.labels['TAGS'][i])

                torch.save(data, self.dir + x.split('.')[0] + '.pt')

        print("Saving complete!")

    def load_audio(self):

        total_audio_datas = []

        for audio_path in self.labels['PATH']:
            audio_sample, sr = torchaudio.load(self.dir + audio_path, num_frames=self.sr * self.audio_max)

            if len(audio_sample) > 1:
                audio_sample = audio_sample.mean(dim=0)

            if sr != self.sr:
                audio_sample = torchaudio.functional.resample(audio_sample, orig_freq=sr, new_freq=self.sr)

            total_audio_datas.append(audio_sample)

        return total_audio_datas

    def __getitem__(self, idx):
        '''
        total_audio_datas = self.load_audio()

        audio_sample = total_audio_datas[idx]
        label = self.labels['TAGS'][idx]
        '''

        audio_path = self.dir + self.labels['PATH'].iloc[idx].split('.')[0] + '.pt'
        audio_sample, label = torch.load(audio_path).values()

        return {"audio" : audio_sample[0],
                "label" : label[0]
        }


class MelSpectrogram(nn.Module):
    def __init__(self,
                 sr=16000,
                 n_fft = 1024,
                 hop_length = 512,
                 n_mels = 48):
        super().__init__()
        self.mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  n_fft=n_fft,
                                                                  hop_length=hop_length,
                                                                  n_mels=n_mels)
    def forward(self, input):
        mel_spec = self.mel_converter(input)

        return mel_spec


def create_audio_data_loader(DATA_DIR, split, num_max_data, batch_size):
    if 'MTAT' in DATA_DIR:
        dataset = OnMemoryDataset(
            DATA_DIR,
            split=split,
            num_max_data=num_max_data
        )

    else:
        dataset = JamendoDataset(
            DATA_DIR,
            split=split,
            num_max_data=num_max_data
        )

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4
    )


if __name__ == "__main__":
    MTAT_DIR = '../dataset/MTAT_SMALL/'
    JAMENDO_DIR = '../dataset/mtg-jamendo-dataset/'
    BATCH_SIZE = 8
    NUM_MAX_DATA = 16
    AUDIO_MAX = 500

    example = JamendoDataset(JAMENDO_DIR, 'train', AUDIO_MAX)
    print(example[4]['audio'].shape)
    print(example[4]['label'])

    mel_spec = MelSpectrogram()
    print(mel_spec(example[4]['audio']).shape)

    '''
    example = OnMemoryDataset(MTAT_DIR, split='train', num_max_data=NUM_MAX_DATA)
    audio, label = example[4]
    print(example.vocab[torch.where(label)])

    train_data_loader = create_audio_data_loader(MTAT_DIR, 'train', NUM_MAX_DATA, BATCH_SIZE)
    example = next(iter(train_data_loader))
    print(example[0].shape, example[1].shape)

    mel_spec = MelSpectrogram()
    print(mel_spec(example[0]).shape)
    '''