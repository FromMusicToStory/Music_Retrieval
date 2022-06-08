import pandas as pd
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


class MelSpectogram(nn.Module):
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


def create_data_loader(MTAT_DIR, split, num_max_data, batch_size):
    dataset = OnMemoryDataset(
        MTAT_DIR,
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
    BATCH_SIZE = 16
    NUM_MAX_DATA = 50

    example = OnMemoryDataset(MTAT_DIR, split='train', num_max_data=50)
    audio, label = example[10]
    print(audio.shape)
    print(label.shape)
    print(example.vocab[torch.where(label)])

    train_data_loader = create_data_loader(MTAT_DIR, 'train', NUM_MAX_DATA, BATCH_SIZE)

    example = next(iter(train_data_loader))
    print(example[0].shape, example[1].shape)

    mel_spec = MelSpectogram()
    print(mel_spec(example[0]).shape)
