import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = torch.add(x, self.pe[:x.size(-2), :])
        return x


# class LearnablePositionalEncoding(nn.Module):

#     def __init__(self, dict_size=128, num_pos_feats=16):
#         super().__init__()
#         self.embed = nn.Embedding(dict_size, num_pos_feats)

#     def forward(self, x):
#         w = x.shape[-2]
#         i = torch.arange(w, device=x.device)
#         emb = self.embed(i)
#         x = torch.add(x, emb)
#         return x