import torch
import torch.nn as nn

from diffusion_policy.model.force_torque.ft_transformer import PositionalEncoding

class EndEffectorEncoder(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=3):
        super(EndEffectorEncoder, self).__init__()

        self.embedding_ee = nn.Linear(6, d_model)

        self.positional_encoding_ee = PositionalEncoding(d_model, max_len=1)

        self.transformer_encoder_ee = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
            num_layers=num_encoder_layers
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, ee_data):

        ee_data = ee_data.unsqueeze(1)
        ee_data = self.embedding_ee(ee_data)

        ee_data = self.positional_encoding_ee(ee_data)

        ee_data = self.transformer_encoder_ee(ee_data)
        ee_data = self.layer_norm(ee_data)

        output = self.fc(ee_data)

        # Flatten to get [batch_size, d_model] shape
        output = output.view(output.size(0), -1)  

        return output
