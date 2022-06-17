import os
import json
import torch
import torch.utils.data as data
from transformers import AutoTokenizer, AutoModel


class StoryTextDataset(data.Dataset):
    def __init__(self, dir_path, split='train', max_len=512):
        self.dir = dir_path
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
        self.max_len = max_len

        if split == 'train':
            self.data = json.load(open(self.dir + "train_filtered_story_eng_label.json", 'r'))
        elif split == 'valid':
            self.data = json.load(open(self.dir + "valid_filtered_story_eng_label.json", 'r'))
        elif split == 'test':
            self.data = json.load(open(self.dir + "test_filtered_story_eng_label.json", 'r'))
        else:
            if os.path.exists(self.dir + "story_dataset.json"):
                self.data = json.load(open(self.dir + "story_dataset.json", 'r'))

            else:
                train = json.load(open(self.dir + "train_filtered_story_eng_label.json", 'r'))
                valid = json.load(open(self.dir + "valid_filtered_story_eng_label.json", 'r'))
                test = json.load(open(self.dir + "test_filtered_story_eng_label.json", 'r'))

                with open(self.dir + "story_dataset.json", "w") as new_file:
                    json.dump(train + valid + test, new_file)

                self.data = json.load(open(self.dir + "story_dataset.json", 'r'))

        self.emotion_map = {
            'happy': 0,
            'neutral': 1,
            'flustered': 2,
            'anxious': 2,
            'angry': 3,
            'sad': 4,
            'hurt': 4
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]['recite_src']['text']
        emotion = self.data[index]['recite_src']['styles'][0]['emotion']
        target = self.emotion_map[emotion]

        tokenized = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding = 'max_length',
            truncation=True,
            return_tensors='pt')

        return {
            'text': text,
            'input_ids': tokenized['input_ids'].flatten(),
            'mask': tokenized['attention_mask'].flatten(),
            'target': torch.tensor(target, dtype=torch.long)
        }


def create_text_data_loader(TEXT_DIR, split, max_len, batch_size):
    dataset = StoryTextDataset(
        TEXT_DIR,
        split=split,
        max_len=max_len
    )

    return data.DataLoader(
        dataset,
        batch_size=batch_size
    )


if __name__ == '__main__':
    TEXT_DIR = '../dataset/Story_dataset/'
    BATCH_SIZE = 16
    MAX_LEN = 512

    dataset = StoryTextDataset(TEXT_DIR, 'train', MAX_LEN)
    train_data_loader = create_text_data_loader(TEXT_DIR, 'none', MAX_LEN, BATCH_SIZE)
    example = next(iter(train_data_loader))

    print(example.keys())
    print(example['input_ids'].shape)
    print(example['mask'].shape)
    print(example['target'].shape)
