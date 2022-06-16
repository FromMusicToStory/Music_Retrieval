import torch
from torch import nn
from text.model import *
from audio.reference_encoder_gst import *
from audio.reference_encoder_vae import *
from metric_embedding_dataloader import *


class MLEmbedModel(nn.Module):
    def __init__(self, ndim, reference_style = 'gst', margin =0.2):
        super(self).__init__()
        self.n_dim  = ndim
        self.text_encoder = TextEncoder(output_dim=self.ndim)

        if reference_style == 'gst':
            self.reference_encoder = ReferenceEncoder(idim=6648)
            self.audio_encoder = StyleTokenLayer()
        else:
            self.reference_encoder = ReferenceEncoder(idim=6648, gru_units=self.ndim*2)
            self.audio_encoder = VAE_StyleTokenLayer(gru_units=self.ndim*2)

        self.relu = nn.ReLU()
        self.margin = margin


    def audio_to_embedding(self, batch):
        ref_embed = self.reference_encoder(batch)
        style_token = self.audio_encoder(ref_embed)

        return style_token


    def text_to_embedding(self, batch):
        positive_embed = self.text_encoder(batch['pos_input_ids'], batch['pos_mask'])
        negative_embed = self.text_encoder(batch['neg_input_ids'], batch['neg_mask'])

        return positive_embed, negative_embed


    def forward(self, batch):
        if batch != None:
            audio_embed = self.audio_to_embedding(batch)
            text_positive_embed, text_negative_embed = self.text_to_embedding(batch)

            cosine_positive = nn.CosineSimilarity(dim=-1)(audio_embed, text_positive_embed)
            cosine_negative = nn.CosineSimilarity(dim=-1)(audio_embed, text_negative_embed)

            losses = self.relu(self.margin - cosine_positive + cosine_negative)
            return losses.mean()


if __name__ == "__main__":
    AUDIO_DIR = '../dataset/mtg-jamendo-dataset/'
    TEXT_DIR = '../dataset/Story_dataset/'
    BATCH_SIZE = 32
    MAX_LEN = 512
    AUDIO_MAX = 500

    test =  AudioTextDataset(AUDIO_DIR , TEXT_DIR, 'valid', MAX_LEN, AUDIO_MAX)
    data_loader = create_data_loader(AUDIO_DIR , TEXT_DIR, 'valid', MAX_LEN, AUDIO_MAX, BATCH_SIZE)

    example = next(iter(data_loader))

    model = MLEmbedModel(ndim=MAX_LEN)
    print(model)