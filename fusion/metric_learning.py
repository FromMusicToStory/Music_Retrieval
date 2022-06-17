import torch
from torch import nn
from text.model import *
from audio.reference_encoder_gst import *
from audio.reference_encoder_vae import *
from metric_embedding_dataloader import *
from sklearn.metrics.pairwise import *
from pytorch_metric_learning import losses


class MLEmbedModel(nn.Module):
    def __init__(self, ndim, reference_style = 'gst', margin =0.2, device ='cpu'):
        super().__init__()
        self.ndim  = ndim
        self.text_encoder = TextEncoder(output_dim=self.ndim)
        self.device = device

        if reference_style == 'gst':
            self.reference_encoder = ReferenceEncoder(idim=489)
            self.audio_encoder = StyleTokenLayer()
        else:
            self.reference_encoder = ReferenceEncoder(idim=489, gru_units=self.ndim*2)
            self.audio_encoder = VAE_StyleTokenLayer(gru_units=self.ndim*2)


        self.cosine = nn.CosineSimilarity(dim=-1)
        self.relu = nn.ReLU()
        self.margin = margin

        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)                               # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
        self.triplet_distance_loss = nn.TripletMarginWithDistanceLoss(distance_function=
                                                                      nn.PairwiseDistance())    # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginWithDistanceLoss.html


    def audio_to_embedding(self, batch):
        anchor = batch['anchor'].to(self.device)
        ref_embed = self.reference_encoder(anchor)
        style_token = self.audio_encoder(ref_embed)

        return style_token


    def text_to_embedding(self, batch):
        pos_input_ids = batch['pos_input_ids'].to(self.device)
        pos_mask = batch['pos_mask'].to(self.device)
        neg_input_ids = batch['neg_input_ids'].to(self.device)
        neg_mask = batch['neg_mask'].to(self.device)

        positive_embed = self.text_encoder(pos_input_ids, pos_mask)
        negative_embed = self.text_encoder(neg_input_ids, neg_mask)

        return positive_embed, negative_embed


    def forward(self, batch):
        if batch != None:
            audio_embed = self.audio_to_embedding(batch)
            text_positive_embed, text_negative_embed = self.text_to_embedding(batch)

            cosine_positive = self.cosine(audio_embed, text_positive_embed)
            cosine_negative = self.cosine(audio_embed, text_negative_embed)

            losses = self.relu(self.margin - cosine_positive + cosine_negative)

            return audio_embed, text_positive_embed, text_negative_embed, losses.mean()


    def evaluate(self, batch):
        audio_embed = self.audio_to_embedding(batch)
        text_positive_embed, text_negative_embed = self.text_to_embedding(batch)

        cosine_positive = nn.CosineSimilarity(dim=-1)(audio_embed, text_positive_embed)
        cosine_negative = nn.CosineSimilarity(dim=-1)(audio_embed, text_negative_embed)

        losses = self.relu(self.margin - cosine_positive + cosine_negative)

        triplet_loss = self.triplet_loss(audio_embed, text_positive_embed, text_negative_embed)
        triplet_distance_loss = self.triplet_distance_loss(audio_embed, text_positive_embed, text_negative_embed)

        audio_embed = audio_embed.cpu().numpy()
        text_positive_embed = text_positive_embed.cpu().numpy()

        cosine_similarity = paired_cosine_distances(audio_embed, text_positive_embed)
        manhattan_distances = paired_manhattan_distances(audio_embed, text_positive_embed)
        euclidean_distances = paired_euclidean_distances(audio_embed, text_positive_embed)

        score = {
            "loss": losses.mean(),
            "triplet_loss": triplet_loss.mean(),
            "triplet_distance_loss": triplet_distance_loss.mean(),
            "cosine_similarity": cosine_similarity.mean(),
            "manhattan_distance": manhattan_distances.mean(),
            "euclidean_distance": euclidean_distances.mean(),
        }

        return score


if __name__ == "__main__":
    AUDIO_DIR = '../dataset/mtg-jamendo-dataset/'
    TEXT_DIR = '../dataset/Story_dataset/'
    BATCH_SIZE = 32
    MAX_LEN = 512
    AUDIO_MAX = 500

    # test =  AudioTextDataset(AUDIO_DIR , TEXT_DIR, 'valid', MAX_LEN, AUDIO_MAX)
    data_loader = create_data_loader(AUDIO_DIR , TEXT_DIR, 'valid', MAX_LEN, AUDIO_MAX, BATCH_SIZE)

    example = next(iter(data_loader))
    
    model = MLEmbedModel(ndim=MAX_LEN)
    print(model(example))
    print(model.evaluate(example))
