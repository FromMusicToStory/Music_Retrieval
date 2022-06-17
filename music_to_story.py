import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from audio.reference_encoder_gst import StyleEncoder, VAE_StyleEncoder
from audio.dataloader import MelDataset
from multi_dataloader import AudioTextDataset
from text.model import TextEncoder
from text.dataloader import TextClassification_Dataset
from word2vec import W2V

import argparse
import tqdm


def create_data_loader(modal, data_dir, split, batch_size, num_workers, **kwargs):
    if modal == 'text':
        dataset = TextClassification_Dataset(data_dir, split, max_len=kwargs['max_len'])
    else:
        if split == 'train':
            dataset = MelDataset(data_dir, split, num_max_data=kwargs['data_num'])
        else :
            dataset = MelDataset(data_dir, split)

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

class MetricModel(nn.Module):
    def __init__(self, style_encoder = 'gst', n_dim=256,out_dim=64):

        if style_encoder == 'gst':
            self.style_encoder = StyleEncoder(idim=15626)
        else :
            self.style_encoder = VAE_StyleEncoder(idim=15626)
        self.text_encoder = TextEncoder()

        # audio MLP
        self.audio_mlp = nn.Sequential(
            nn.Linear(n_dim, n_dim*2),
            nn.BatchNorm1d(n_dim*2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_dim*2, out_dim)
        )
        # text MLP
        self.text_mlp = nn.Sequential(
            nn.Linear(512, n_dim*2),
            nn.BatchNorm1d(n_dim*2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_dim*2, out_dim)
        )
        # tag MLP
        self.tag_mlp = nn.Sequential(
            nn.Linear(300, n_dim *2),
            nn.BatchNorm1d(n_dim*2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_dim*2, out_dim)
        )

        self.loss_func= nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity())

    def forward(self, **inputs):
        # text_tag, audio_tag, spec, text, neg_spec, neg_text
        spec = self.style_encoder(inputs['mel'])
        text = self.text_encoder(inputs['text'])

        text_tag_emb = self.tag_mlp(inputs['text_label'])
        audio_tag_emb = self.tag_mlp(inputs['mel_label'])
        audio_emb = self.audio_mlp(spec)
        text_emb = self.text_mlp(text)

        neg_spec = self.style_encoder(inputs['neg_mel'])
        neg_text = self.text_encoder(inputs['neg_text'])

        neg_spec_emb = self.audio_mlp(neg_spec)
        neg_text_emb = self.text_mlp(neg_text)

        loss = self.loss_func(text_tag_emb, text_emb,neg_text_emb)
        loss += self.loss_func(audio_tag_emb, audio_emb, neg_spec_emb)
        loss += self.loss_func(audio_emb, text_emb, neg_text_emb)

        return loss.mean()


import os

def main():
    os.environ['TOKENIZERS_PARALLELISM']='true'
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_dir', type=str, default='dataset/Story_dataset/')
    parser.add_argument('--text_batch_size', type=int, default=64)
    parser.add_argument('--text_max_len', type=int, default=512)
    parser.add_argument('--audio_dir', type=str, default='dataset/MTG/')
    parser.add_argument('--audio_batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--ckp_per_step', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load data
    train_dataset = AudioTextDataset(audio_dir=args.audio_dir, text_dir=args.text_dir,split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    valid_dataset = AudioTextDataset(audio_dir=args.audio_dir, text_dir=args.text_dir,split='valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=64)

    # model
    model = MetricModel()

    if device=='cuda':
        model = model.cuda(device)

    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model = model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1



    for epoch in range(start_epoch, args.epoch):
        loss = train(model, train_dataloader)
        val_loss = validate(model, valid_dataloader)
        print(f'{epoch} done')
        print('start saving checkpoint')
        ckp_path = os.path.join(args.checkpoint_dir, f'metric_{epoch}_val_loss_{val_loss:.5f}.pth')
        torch.save({'epoch':epoch, 'state_dict':model.state_dict()}, ckp_path)



def train(model, dataloader, ): # for one epoch
    model.train()

    avg_loss = 0
    for idx, batch in tqdm(enumerate(dataloader)):
        input = batch
        output = model(**input)

        output.backward()

    return output

def validate(model, dataloader,):
    model.eval()

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader)):
            output = model(**batch)


            return score



if __name__=='__main__':
    main()
