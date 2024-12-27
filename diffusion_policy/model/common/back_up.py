from torch.nn.modules.activation import MultiheadAttention
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
import torch
from torch import nn 

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.force_torque.end_effector_encoding import EndEffectorEncoder
from diffusion_policy.model.force_torque.ft_transformer import ForceTorqueEncoder


class Actor(ModuleAttrMixin):
    def __init__(self, encoder_dim =256, num_heads = 8, action_dim =6, use_eef_encoder = True):
        super().__init__()

        self.force_torque_encoder = ForceTorqueEncoder(ft_seq_len=10).to(self.device)
        if use_eef_encoder:
            self.end_effector_encoder = EndEffectorEncoder().to(self.device)
        else:
            self.end_effector_encoder = nn.Linear(6, encoder_dim)

        

        self.layernorm_embed_shape = encoder_dim 
        self.encoder_dim = encoder_dim
        
        self.use_mha = True

        self.modalities = ['force_torque', 'end_effector']  

        self.embed_dim = self.layernorm_embed_shape * len(self.modalities)


        self.layernorm = nn.LayerNorm(self.layernorm_embed_shape)
        self.mha = MultiheadAttention(self.layernorm_embed_shape, num_heads)
        self.bottleneck = nn.Linear(
            self.embed_dim, self.layernorm_embed_shape
        )  # if we dont use mha



        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.layernorm_embed_shape, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 3**action_dim),
        )
        self.aux_mlp = torch.nn.Linear(self.layernorm_embed_shape, 6)

    def forward(self, ft_data,end_effector):
        """
        Args:
        
            ft_data: [batch, dim]
            end_effector: [batch, dim]

        """
        
        embeds = []

        ft_data = self.force_torque_encoder(ft_data)
        ft_data = ft_data.view(-1, self.layernorm_embed_shape)
        embeds.append(ft_data)

        end_effector = self.end_effector_encoder(end_effector)
        end_effector = end_effector.view(-1, self.layernorm_embed_shape)
        embeds.append(end_effector)

    

 
        mlp_inp = torch.stack(embeds, dim=0)  # [3, batch, D]

        mha_out, weights = self.mha(mlp_inp, mlp_inp, mlp_inp)  # [1, batch, D]
        mha_out += mlp_inp
        mlp_inp = torch.concat([mha_out[i] for i in range(mha_out.shape[0])], 1)
        mlp_inp = self.bottleneck(mlp_inp)




        action_logits = self.mlp(mlp_inp)
        xyzrpy = self.aux_mlp(mlp_inp)
        return action_logits,xyzrpy , weights
