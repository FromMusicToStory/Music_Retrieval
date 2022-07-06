import torch
import torch.nn as nn

import math
from typing import Sequence


# reference encoder from GST
# code from espnet

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
                 in_dim = 256,
                 out_dim = 64,
                 style_layer ='gst'
                ):
        super(StyleEncoder, self).__init__()

        self.ref_enc = ReferenceEncoder(idim = idim,
                                        conv_layers = conv_layers,
                                        conv_channels_list= conv_channels_list,
                                        conv_kernel_size = conv_kernel_size,
                                        conv_stride= conv_stride,
                                        gru_layers = gru_layers,
                                        gru_units = gru_units)

        if style_layer =='gst':
            self.style_token_layer = StyleTokenLayer(ref_embed_dim = gru_units,
                                                 gst_tokens = gst_tokens,
                                                 gst_token_dim = gst_token_dim,
                                                 gst_heads =gst_heads,
                                                 )
        else:
            self.audio_encoder = VAE_StyleTokenLayer(gru_units=gru_units)






    def forward(self, speech: torch.Tensor) -> torch.Tensor:

        ref_embeds = self.ref_enc(speech)
        style_embeds = self.style_token_layer(ref_embeds)

        return style_embeds

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
            convs += [nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size=conv_kernel_size,
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

# GST style token layer
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

''' 
@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson {Enrique Yalta Soplin} and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={{ESPnet}: End-to-End Speech Processing Toolkit},
  year={2018},
  booktitle={Proceedings of Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1456}
}
@inproceedings{hayashi2020espnet,
  title={{Espnet-TTS}: Unified, reproducible, and integratable open source end-to-end text-to-speech toolkit},
  author={Hayashi, Tomoki and Yamamoto, Ryuichi and Inoue, Katsuki and Yoshimura, Takenori and Watanabe, Shinji and Toda, Tomoki and Takeda, Kazuya and Zhang, Yu and Tan, Xu},
  booktitle={Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7654--7658},
  year={2020},
  organization={IEEE}
}
@inproceedings{inaguma-etal-2020-espnet,
    title = "{ESP}net-{ST}: All-in-One Speech Translation Toolkit",
    author = "Inaguma, Hirofumi  and
      Kiyono, Shun  and
      Duh, Kevin  and
      Karita, Shigeki  and
      Yalta, Nelson  and
      Hayashi, Tomoki  and
      Watanabe, Shinji",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-demos.34",
    pages = "302--311",
}
@inproceedings{li2020espnet,
  title={{ESPnet-SE}: End-to-End Speech Enhancement and Separation Toolkit Designed for {ASR} Integration},
  author={Chenda Li and Jing Shi and Wangyou Zhang and Aswin Shanmugam Subramanian and Xuankai Chang and Naoyuki Kamo and Moto Hira and Tomoki Hayashi and Christoph Boeddeker and Zhuo Chen and Shinji Watanabe},
  booktitle={Proceedings of IEEE Spoken Language Technology Workshop (SLT)},
  pages={785--792},
  year={2021},
  organization={IEEE},
}
@article{arora2021espnet,
  title={ESPnet-SLU: Advancing Spoken Language Understanding through ESPnet},
  author={Arora, Siddhant and Dalmia, Siddharth and Denisov, Pavel and Chang, Xuankai and Ueda, Yushi and Peng, Yifan and Zhang, Yuekai and Kumar, Sujay and Ganesan, Karthik and Yan, Brian and others},
  journal={arXiv preprint arXiv:2111.14706},
  year={2021}
}
'''
