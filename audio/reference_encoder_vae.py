import torch
import torch.nn as nn
from reference_encoder_gst import ReferenceEncoder

# reference encoder from tacotron2-vae

class VAE_StyleTokenLayer(nn.Module):
    def __init__(self, gru_units = 128, z_latent_dim = 32):
        super().__init__()
        output_dim = gru_units // 2

        self.fc1 = nn.Linear(gru_units, z_latent_dim)
        self.fc2 = nn.Linear(gru_units, z_latent_dim)
        self.fc_final = nn.Linear(z_latent_dim, output_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, ref_embeddings):
        mu = self.fc1(ref_embeddings)
        logvar = self.fc2(ref_embeddings)
        z = self.reparameterize(mu, logvar)
        style_embed = self.fc_final(z)

        return style_embed, mu, logvar, z


if __name__ == "__main__":
    from dataloader import *

    MTAT_DIR = '../dataset/MTAT_SMALL/'
    BATCH_SIZE = 16
    NUM_MAX_DATA = 50

    train_data_loader = create_data_loader(MTAT_DIR, 'train', NUM_MAX_DATA, BATCH_SIZE)
    example = next(iter(train_data_loader))
    print(example[0].shape)

    mel_spec = MelSpectrogram()
    mel_spec_out = mel_spec(example[0])
    print(mel_spec_out.shape)

    reference_encoder = ReferenceEncoder(idim=911, gru_units=1024)
    ref_embed = reference_encoder(mel_spec_out)
    print(ref_embed.shape)

    style_token = VAE_StyleTokenLayer(gru_units=1024)
    print(style_token(ref_embed)[0].shape)