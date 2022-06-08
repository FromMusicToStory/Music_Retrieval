import torch
import torch.nn as nn
from reference_encoder_gst import ReferenceEncoder

# reference encoder from tacotron2-vae

class VAE_StyleTokenLayer(nn.Module):
    def __init__(self, gru_units = 128, z_latent_dim = 32):
        super().__init__()
        output_dim = gru_units // 2

        self.ref_encoder = ReferenceEncoder()
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

    def forward(self, inputs):
        encoder_out = self.ref_encoder(inputs)
        mu = self.fc1(encoder_out)
        logvar = self.fc2(encoder_out)
        z = self.reparameterize(mu, logvar)
        style_embed = self.fc_final(z)

        return style_embed, mu, logvar, z


