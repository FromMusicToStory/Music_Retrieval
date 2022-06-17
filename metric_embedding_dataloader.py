from audio.dataloader import *
from text.dataloader import *
from match_label import make_label
import random


class AudioTextDataset(data.Dataset):
    def __init__(self, audio_dir, text_dir, split, max_len, audio_max):
        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self.split = split
        self.max_len = max_len
        self.audio_max = audio_max

        self.audio_dataset = JamendoDataset(self.audio_dir, self.split, self.audio_max)
        self.text_dataset = StoryTextDataset(self.text_dir, 'none', self.max_len)
        self.mel_spec = MelSpectrogram()


    def get_text_tag_from_audio(self, tag, select='positive'):
        label = make_label()
        if select == 'positive':
            if tag in label['happy']['positive']:
                return 0
            if tag in label['neutral']['positive']:
                return 1
            if tag in label['flustered']['positive']:
                return 2
            if tag in label['angry']['positive']:
                return 3
            else:
                return 4

        else:
            if tag in label['happy']['negative']:
                return random.sample([2, 3, 4], 1)[0]
            if tag in label['neutral']['negative']:
                return random.sample([0, 2, 3, 4], 1)[0]
            if tag in label['flustered']['negative']:
                return random.sample([0, 1], 1)[0]
            if tag in label['angry']['negative']:
                return random.sample([0, 1, 4], 1)[0]
            else:
                return random.sample([0, 3], 1)[0]


    def get_random_text(self, tag, select='positive'):
        random_number = random.randint(0, len(self.text_dataset))
        tag_num  = self.get_text_tag_from_audio(tag, select=select)

        if self.text_dataset[random_number]['target'] == tag_num:
            return self.text_dataset[random_number]

        else:
            while self.text_dataset[random_number]['target'] != tag_num:
                random_number = random.randint(0, len(self.text_dataset))
                return self.text_dataset[random_number]

    def __len__(self):
        return len(self.audio_dataset)

    def __getitem__(self, idx):
        audio = self.audio_dataset[idx]
        audio_tag = audio['label']

        if len(audio_tag) > 1:
            for i in range(len(audio_tag)):
                result = []
                result.append(self.get_text_tag_from_audio(audio_tag[i], select='positive'))
                if len(set(result)) != 1:
                    audio_tag = audio_tag[0]
                else:
                    return

        pos_text = self.get_random_text(audio_tag, select='positive')
        neg_text = self.get_random_text(audio_tag, select='negative')


        return {
            "anchor" : self.mel_spec(audio['audio']),
            "pos_input_ids": pos_text['input_ids'],
            "pos_mask" : pos_text['mask'],
            "neg_input_ids": neg_text['input_ids'],
            "neg_mask" : neg_text['mask']
        }


def create_data_loader(AUDIO_DIR , TEXT_DIR, split, max_len, audio_max, batch_size):
    dataset = AudioTextDataset(
        AUDIO_DIR, TEXT_DIR,
        split=split,
        max_len=max_len,
        audio_max = audio_max
    )

    return data.DataLoader(
        dataset,
        batch_size=batch_size
    )


if __name__ == "__main__":
    AUDIO_DIR = 'dataset/mtg-jamendo-dataset/'
    TEXT_DIR = 'dataset/Story_dataset/'
    BATCH_SIZE = 32
    MAX_LEN = 512
    AUDIO_MAX = 500

    test =  AudioTextDataset(AUDIO_DIR , TEXT_DIR, 'test', MAX_LEN, AUDIO_MAX)
    print(test[0]['anchor'].shape)
    print(test[0]['pos_input_ids'].shape)
    print(test[0]['neg_input_ids'].shape)

    data_loader = create_data_loader(AUDIO_DIR , TEXT_DIR, 'valid', MAX_LEN, AUDIO_MAX, BATCH_SIZE)
    example = next(iter(data_loader))
    print(example['anchor'].shape)
    print(example['pos_input_ids'].shape)
    print(example['neg_input_ids'].shape)