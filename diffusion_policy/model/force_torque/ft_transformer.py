import torch
import torch.nn as nn
 
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, model_dim)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
  
 
class ForceTorqueEncoder(nn.Module):
 
    def __init__(self, ft_seq_len, d_model=256, nhead=8, num_encoder_layers=3):
        super(ForceTorqueEncoder, self).__init__()
 
        self.ft_seq_len = ft_seq_len

        # input embedding layer
        self.embedding_ft = nn.Linear(6, d_model)   # batch_size, seq_len, 6 -> batch_size, seq_len, d_model

        self.positional_encoding_ft = PositionalEncoding(d_model, max_len=ft_seq_len) # batch_size, seq_len, d_model -> batch_size, seq_len, d_model

        self.transformer_encoder_ft = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True ),
            num_layers=num_encoder_layers
        ) # batch_size, seq_len, d_model -> batch_size, seq_len, d_model

        self.layer_norm = nn.LayerNorm(d_model) 

        self.fc = nn.Linear(d_model * ft_seq_len, d_model) 
 
    
    def forward(self, ft_data):
        
        ft_data = self.embedding_ft(ft_data)
        ft_data = self.positional_encoding_ft(ft_data)
        ft_data = self.transformer_encoder_ft(ft_data)

        ft_data = self.layer_norm(ft_data)

        output = ft_data.view(ft_data.size(0), -1)  
        output = self.fc(output) 



        return output