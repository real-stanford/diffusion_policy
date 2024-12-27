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

        self.force_torque_encoder = ForceTorqueEncoder(ft_seq_len=10)
        if use_eef_encoder:
            self.end_effector_encoder = EndEffectorEncoder()
        else:
            self.end_effector_encoder = nn.Linear(6, encoder_dim)

        

        self.layernorm_embed_shape = encoder_dim 
        self.encoder_dim = encoder_dim
        
        self.use_mha = True

        self.modalities = ['force_torque',"cf0" ]  

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

    def forward(self, ft_data, cf0= None,cf1=None,cf2=None,cf3=None ):
        """
        Args:
        
            ft_data: [batch, dim]
            end_effector: [batch, dim]

        """
        
        embeds = []

        ft_data = self.force_torque_encoder(ft_data)
        ft_data = ft_data.view(-1, self.layernorm_embed_shape)
        embeds.append(ft_data)

        # end_effector = self.end_effector_encoder(end_effector)
        # end_effector = end_effector.view(-1, self.layernorm_embed_shape)
        # embeds.append(end_effector)

        if cf0 is not None:
            cf0 = cf0.view(-1, self.layernorm_embed_shape)
            embeds.append(cf0)
        if cf1 is not None:
            cf1 = cf1.view(-1, self.layernorm_embed_shape)
            embeds.append(cf1)
        if cf2 is not None:
            cf2 = cf2.view(-1, self.layernorm_embed_shape)
            embeds.append(cf2)
        if cf3 is not None:
            cf3 = cf3.view(-1, self.layernorm_embed_shape)
            embeds.append(cf3)
        

    

 
        mlp_inp = torch.stack(embeds, dim=0)  # [3, batch, D]

        mha_out, weights = self.mha(mlp_inp, mlp_inp, mlp_inp)  # [1, batch, D]
        mha_out += mlp_inp
        mlp_inp = torch.concat([mha_out[i] for i in range(mha_out.shape[0])], 1)
        mlp_inp = self.bottleneck(mlp_inp)




        action_logits = self.mlp(mlp_inp)
        xyzrpy = self.aux_mlp(mlp_inp)
        return action_logits,xyzrpy , weights
