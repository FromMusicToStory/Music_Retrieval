import json
import torch
import torch.utils.data as data
from transformers import AutoTokenizer, AutoModel


class TextClassification_Dataset(data.Dataset):
    def __init__(self, dir_path, split='train', max_len=512):
        self.dir = dir_path
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
        self.max_len = max_len

        if split == 'train':
            self.data = json.load(open(self.dir + "train_filtered_story_eng_label.json", 'r'))
        elif split == 'valid':
            self.data = json.load(open(self.dir + "valid_filtered_story_eng_label.json", 'r'))
        else:
            self.data = json.load(open(self.dir + "test_filtered_story_eng_label.json", 'r'))

        self.emotion_map = {
            'happy': 0,
            'flustered': 1,
            'neutral': 2,
            'angry': 3,
            'anxious': 4,
            'hurt': 5,
            'sad': 6
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]['recite_src']['text']
        emotion = self.data[index]['recite_src']['styles'][0]['emotion']

        tokenized = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            return_tensors='pt')

        return {
            'text': text,
            'input_ids': tokenized['input_ids'].flatten(),
            'mask': tokenized['attention_mask'].flatten(),
            'targets': torch.tensor(self.emotion_map[emotion], dtype=torch.long)
        }


def create_data_loader(TEXT_DIR, split, max_len, batch_size):
    dataset = TextClassification_Dataset(
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
    train_data_loader = create_data_loader(TEXT_DIR, 'train', MAX_LEN, BATCH_SIZE)

    example = next(iter(train_data_loader))

    print(example.keys(), example.values())
    print(example['input_ids'].shape)
    print(example['mask'].shape)
    print(example['targets'].shape)