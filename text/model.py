from transformers import AutoModel
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, freeze=False, input_dim=768, output_dim=512, num_label=7):
        super().__init__()
        self.model = AutoModel.from_pretrained("beomi/KcELECTRA-base", return_dict=True)

        if freeze:
            for name, params in self.model.named_parameters():
                params.required_grad = False

        self.projection = nn.Linear(input_dim, output_dim)
        self.clf = nn.Linear(output_dim, num_label)

    def forward(self, input_ids, attention_mask, return_hidden_states=False, do_clf=False):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask)

        last_hidden_state = output.last_hidden_state

        if return_hidden_states:
            return last_hidden_state            # batch, max_len, hidden_size(768)

        x = last_hidden_state[:, 0, :]          # batch, max_len, hidden_size
        x = self.projection(x)                  # batch, output_dim

        if do_clf:
            x = self.clf(x)                     # batch, num_label

        return x



if __name__ == "__main__":
    from dataloader import *

    TEXT_DIR = '../dataset/Story_dataset/'
    BATCH_SIZE = 16
    MAX_LEN = 512

    train_data_loader = create_text_data_loader(TEXT_DIR, 'train', MAX_LEN, BATCH_SIZE)

    example = next(iter(train_data_loader))
    input_ids = example['input_ids']
    mask = example['mask']

    model = TextEncoder()
    print(model(input_ids, mask).shape)
