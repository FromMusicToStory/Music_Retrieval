import torch
import torch.nn as nn

from sklearn.metrics.pairwise import *

from model.audio import StyleEncoder
from model.text import TextEncoder

class TwoBranchMetricModel(nn.Module):
    def __init__(self, ndim, reference_style = 'gst', margin =0.2, device ='cpu'):
        super().__init__()
        self.ndim  = ndim
        self.device = device

        if reference_style =='gst':
            self.style_encoder = StyleEncoder(idim=489, style_layer=reference_style)
        elif reference_style =='vae':
            self.style_encoder = StyleEncoder(idim=489, style_layer=reference_style,gru_units=self.ndim*2)
        self.text_encoder = TextEncoder(output_dim = self.ndim)



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


    def text_to_embedding_only(self, input_ids, atten_mask):
        input_ids = input_ids.to(self.device)
        atten_mask = atten_mask.to(self.device)
        embed = self.text_encoder(input_ids, atten_mask)

        return embed


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

        return

class ThreeBranchMetricModel(nn.Module):
    def __init__(self, reference_style='gst', idim=15626, n_dim=256, out_dim=64):
        super(ThreeBranchMetricModel, self).__init__()


        self.style_encoder = StyleEncoder(idim=idim, style_layer=reference_style)
        self.text_encoder = TextEncoder()

        # audio MLP
        self.audio_mlp = nn.Sequential(
            nn.Linear(n_dim, n_dim * 2),
            nn.BatchNorm1d(n_dim * 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_dim * 2, out_dim)
        )

        # text MLP
        self.text_mlp = nn.Sequential(
            nn.Linear(512, n_dim * 2),
            nn.BatchNorm1d(n_dim * 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_dim * 2, out_dim)
        )
        # tag MLP
        self.tag_mlp = nn.Sequential(
            nn.Linear(300, n_dim * 2),
            nn.BatchNorm1d(n_dim * 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_dim * 2, out_dim)
        )

        self.loss_func = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity())

    def forward(self, batch):
        # text_tag, audio_tag, spec, text, neg_spec, neg_text
        spec = self.style_encoder(batch['mel'])
        text = self.text_encoder(batch['text']['input_ids'], batch['text']['attention_mask'])

        text_tag_emb = self.tag_mlp(batch['text_label'])
        audio_tag_emb = self.tag_mlp(batch['mel_label'])
        audio_emb = self.audio_mlp(spec)
        text_emb = self.text_mlp(text)

        neg_spec = self.style_encoder(batch['neg_mel'])
        neg_text = self.text_encoder(batch['neg_text']['input_ids'], batch['neg_text']['attention_mask'])

        neg_spec_emb = self.audio_mlp(neg_spec)
        neg_text_emb = self.text_mlp(neg_text)

        loss = self.loss_func(text_tag_emb, text_emb, neg_text_emb)
        loss += self.loss_func(audio_tag_emb, audio_emb, neg_spec_emb)
        loss += self.loss_func(audio_emb, text_emb, neg_text_emb)

        return loss.mean()

    def audio_to_embedding(self, batch):
        emb = self.style_encoder(batch['mel'])
        emb = self.audio_mlp(emb)
        return emb

    def text_to_embedding(self, batch):
        pos = self.text_encoder(batch['text']['input_ids'], batch['text']['attention_mask'])
        pos = self.text_mlp(pos)
        neg = self.text_encoder(batch['neg_text']['input_ids'], batch['neg_text']['attention_mask'])
        neg = self.text_mlp(neg)
        return pos, neg

    def text_to_embedding_only(self, input_ids, attention_mask):
        embeds = self.text_encoder(input_ids, attention_mask)
        embeds = self.text_mlp(embeds)
        return embeds

    def evaluate(self, batch):
        audio_embed = self.audio_to_embedding(batch)
        text_positive_embed, text_negative_embed = self.text_to_embedding(batch)

        loss = self.loss_func(audio_embed, text_positive_embed, text_negative_embed)
        marginloss_func = nn.TripletMarginLoss()
        triplet_loss = marginloss_func(audio_embed, text_positive_embed, text_negative_embed)

        audio_embed = audio_embed.cpu().numpy()
        text_positive_embed = text_positive_embed.cpu().numpy()

        cosine_similarity = paired_cosine_distances(audio_embed, text_positive_embed)
        manhattan_distances = paired_manhattan_distances(audio_embed, text_positive_embed)
        euclidean_distances = paired_euclidean_distances(audio_embed, text_positive_embed)

        score = {
            "loss": loss.mean(),
            "triplet_loss": triplet_loss.mean(),
            "triplet_distance_loss": loss.mean(),
            "cosine_similarity": cosine_similarity.mean(),
            "manhattan_distance": manhattan_distances.mean(),
            "euclidean_distance": euclidean_distances.mean(),
        }
        return score