import torch
import torch.nn as nn
import numpy
import math
from typing import Sequence
from torchaudio.transforms import MelSpectrogram

# reference encoder from GST

class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, n_head, n_feat, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        assert n_feat % n_head ==0

        self.dim = n_feat//n_head # dim of q, k, v, they have to be same
        self.head = n_head
        self.linear_q = nn.Linear(q_dim, n_feat)
        self.linear_k = nn.Linear(k_dim, n_feat)
        self.linear_v = nn.Linear(v_dim, n_feat)
        self.linear_output = nn.Linear(n_feat, n_feat)
        self.attention = None
        self.dropout = nn.Dropout(p = dropout_rate)

    def forward(self, q, k, v):

        # first, transform q, k v
        n_batch = q.size(0)
        q = self.linear_q(q).view(n_batch , -1, self.head , self.dim)
        k = self.linear_k(k).view(n_batch, -1, self.head, self.dim)
        v = self.linear_v(v).view(n_batch, -1, self.head, self.dim)

        q = q.transpose(1, 2) # now, (batch, head, time, dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # score
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.dim)
        # now calculate attention context vector
        # there is no mask in this MHA
        self.attention = torch.softmax(scores, dim=-1)

        p_attention = self.dropout(self.attention)

        x = torch.matmul(p_attention, v) # context vector
        x = torch.transpose(x,1,2).contiguous().view(n_batch, -1, self.head * self.dim)

        return self.linear_output(x)


class StyleEncoder(nn.Module):
    def __init__(self,
                 idim: int = 80,
                 gst_tokens :int =10,
                 gst_token_dim : int =256,
                 gst_heads: int = 4,
                 conv_layers: int = 6,
                 conv_channels_list : Sequence[int] = (32,32,64,64,128,128),
                 conv_kernel_size: int =3,
                 conv_stride: int =2,
                 gru_layers: int =1,
                 gru_units: int =128,
                ):
        super(StyleEncoder, self).__init__()

        self.ref_enc = ReferenceEncoder(idim = idim,
                                        conv_layers = conv_layers,
                                        conv_channels_list= conv_channels_list,
                                        conv_kernel_size = conv_kernel_size,
                                        conv_stride= conv_stride,
                                        gru_layers = gru_layers,
                                        gru_units = gru_units)


        self.style_token_layer = StyleTokenLayer(ref_embed_dim = gru_units,
                                                 gst_tokens = gst_tokens,
                                                 gst_token_dim = gst_token_dim,
                                                 gst_heads =gst_heads,
                                                 )

    def forward(self, speech: torch.Tensor) -> torch.Tensor:

        ref_embeds = self.ref_enc(speech)
        style_embeds = self.style_token_layer(ref_embeds)

        return style_embeds


class ReferenceEncoder(nn.Module):
    def __init__(self,
                 idim=80, # dimension of mel-spectrogram
                 conv_layers: int = 6,
                 conv_channels_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
                 conv_kernel_size: int = 3,
                 conv_stride: int = 2,
                 gru_layers: int = 1,
                 gru_units: int = 128,
                 ):
        super(ReferenceEncoder,self).__init__()

        assert conv_kernel_size % 2 == 1, "kernel size must be odd."
        assert len(conv_channels_list) == conv_layers, "the number of conv layers and length of channels list must be the same"
        convs=[]
        padding = (conv_kernel_size-1)//2
        for i in range(conv_layers):
            conv_in_channels = 1 if i == 0 else conv_channels_list[i-1]
            conv_out_channels = conv_channels_list[i]
            convs += [ nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size=conv_kernel_size,
                                 stride = conv_stride, padding= padding, bias = False), # use BatchNorm instead of bias
                       nn.BatchNorm2d(conv_out_channels),
                       nn.ReLU(inplace=True)]
        self.convs = nn.Sequential(*convs)

        self.conv_layers = conv_layers
        self.kernel_size = conv_kernel_size
        self.stride = conv_stride
        self.padding = padding

        gru_in_units = idim
        for i in range(conv_layers):
            gru_in_units = (gru_in_units - conv_kernel_size + 2 *padding) // conv_stride +1
        gru_in_units *= conv_out_channels
        self.gru = nn.GRU(gru_in_units, gru_units, gru_layers, batch_first=True)

    def forward(self, speech: torch.Tensor) -> torch.Tensor:

        batch_size = speech.size(0)
        xs = speech.unsqueeze(1)  # (batch, 1, max_len, idim)
        hs = self.convs(xs).transpose(1,2) # (batch, max_len', conv_out_cahnnels, idim')
        time_length = hs.size(1)
        hs = hs.contiguous().view(batch_size, time_length, -1) # (batch, max_len', gru_units)
        self.gru.flatten_parameters()
        _, ref_embeddings = self.gru(hs)  # (gru_layers, batch_size, gru_units)
        ref_embeddings = ref_embeddings[-1]  # (batch_size, gru_units)

        return ref_embeddings


class StyleTokenLayer(nn.Module):
    def __init__(
            self,
            ref_embed_dim: int = 128,
            gst_tokens: int = 10,
            gst_token_dim: int = 512,
            gst_heads: int = 4,
            dropout_rate: float = 0.0,
    ):
        super(StyleTokenLayer, self).__init__()
        # assert gst_token_dim // gst_heads == ref_embed_dim

        gst_embeddings = torch.randn(gst_tokens, gst_token_dim // gst_heads)
        self.register_parameter("gst_embeddings", nn.Parameter(gst_embeddings))
        self.mha = MultiHeadAttention(
            q_dim=ref_embed_dim,
            k_dim=gst_token_dim // gst_heads,
            v_dim=gst_token_dim // gst_heads,
            n_head=gst_heads,
            n_feat=gst_token_dim,
            dropout_rate=dropout_rate
        )

    def forward(self, ref_embeddings : torch.Tensor) -> torch.Tensor:
        batch_size = ref_embeddings.size(0)
        gst_embeddings = torch.tanh(self.gst_embeddings).unsqueeze(0).expand(batch_size, -1, -1) # batch, gst_tokens, gst_token_dim//gst_heads
        ref_embeddings = ref_embeddings.unsqueeze(1)                                             # batch, 1, ref_embed_dim

        style_embeddings = self.mha(ref_embeddings, gst_embeddings, gst_embeddings)

        return style_embeddings.squeeze(1)


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

    reference_encoder = ReferenceEncoder(idim=911)
    ref_embed = reference_encoder(mel_spec_out)
    print(ref_embed.shape)

    style_token = StyleTokenLayer()
    print(style_token(ref_embed).shape)
