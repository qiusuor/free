import torch
import pickle
import pandas as pd
import numpy as np
import tensorboard as tb
import torch.nn as nn
from torch.nn import LSTMCell
import torch.nn.functional as F
import math



class TransformerEncoder(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, feedforward_dim):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear_1 = torch.nn.Linear(embed_dim, feedforward_dim)
        self.linear_2 = torch.nn.Linear(feedforward_dim, embed_dim)
        self.layernorm_1 = torch.nn.LayerNorm(embed_dim)
        self.layernorm_2 = torch.nn.LayerNorm(embed_dim)
    
    def forward(self, x_in):
        attn_out, _ = self.attn(x_in, x_in, x_in)
        x = self.layernorm_1(x_in + attn_out)
        ff_out = self.linear_2(torch.nn.functional.relu(self.linear_1(x)))
        x = self.layernorm_2(x + ff_out)
        return x
    
class TransformerAutoEncoder(torch.nn.Module):
    def __init__(self, t_len, input_size, lat_size, num_heads=4, dropout=0.05, embed_dim=128, feedforward_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.t_len = t_len
        
        self.pre = nn.Linear(input_size, embed_dim)

        self.encoder = nn.Sequential(
            TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim),
            # TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim),
            # TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim)
        )
        self.h2l = nn.Sequential(
            nn.Linear(embed_dim*t_len, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, lat_size)
        )
        self.l2h = nn.Sequential(
            nn.Linear(lat_size, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim*t_len)
        )
        self.decoder = nn.Sequential(
            TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim),
            # TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim),
            # TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim)
        )
        self.post = nn.Linear(embed_dim, input_size)
        self.position_emb = self.positionalencoding1d(embed_dim, t_len)

    def forward(self, x):
        N = x.shape[0]
        x = self.pre(x)
        x += self.position_emb
        h = self.encoder(x)
        l = self.h2l(h.reshape(N, -1))
        h = self.l2h(l).reshape(N, self.t_len, -1)
        x = self.decoder(h)
        x -= self.position_emb
        x = self.post(x)
        return x, l
    
    def positionalencoding1d(self, d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe.requires_grad = False
        return pe.cuda()
        
  
class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size, output_size, lat_size, hidden_size=16, n_class=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.pre = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.encoder = LSTMCell(hidden_size, hidden_size)
        self.h2l = nn.Sequential(
            # nn.Linear(input_size, hidden_size),
            # nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # self.e2l = nn.Sequential(
        #     nn.Linear(self.D*self.num_layers*2*hidden_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_size, lat_size),
        #     nn.LeakyReLU(),
        # )
        # self.l2c =  nn.Sequential(
        #     nn.Linear(lat_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_size, self.D*self.num_layers*2*hidden_size),
        #     nn.LeakyReLU(),
        # )
        # self.get_start_tocken = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        # )
        self.decoder = LSTMCell(hidden_size, hidden_size)
        self.post = nn.Linear(hidden_size, output_size)
        
        # self.start_tocken = torch.
        # self.classifier = nn.Sequential(
        #     nn.Linear(lat_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_size, n_class),
        # )
        

    def forward(self, seq, device):
        B, T, D = seq.size()
        outputs = torch.zeros_like(seq)

        h, c = torch.zeros(B, self.hidden_size).to(device), torch.zeros(B, self.hidden_size).to(device)
        seq = self.pre(seq)
        start = seq[:,-1,:]
        for i in range(T):
            h, c = self.encoder(seq[:,i,:], (h, c))
        for i in range(T):
            h, c = self.encoder(start, (h, c))
            start = self.h2l(h)
            output = self.post(h)
            outputs[:,-1-i,:] = output
            # outputs.append(output)
        
        # outputs = outputs[::-1]
        # outputs = torch.cat(outputs).transpose()
        # print(seq.shape)
        # _, (h, c) = self.encoder(o)
        # lat = self.e2l(torch.cat([h, c], dim=0).reshape(N, -1))
        # o = self.l2c(lat)
        # start_tocken = self.get_start_tocken(seq)
        # o = torch.cat([h, c], dim=0).reshape(N, -1)
        # o, (h, c) = self.decoder(start_tocken, (o[:,:self.D*self.num_layers*self.hidden_size].reshape(-1, N, self.hidden_size), o[:,self.D*self.num_layers*self.hidden_size:].reshape(-1, N, self.hidden_size)))
        # o = self.post(o)
        # y = self.classifier(lat)
        
        return outputs, outputs
    
class MLPAutoEncoder(nn.Module):
    def __init__(self, input_size, lat_size, hidden_size=128):
        super().__init__()
        self.input_size = input_size
        self.lat_size = lat_size
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, lat_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(lat_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, input_size),
        )
        

    def forward(self, x):
        lat = self.encoder(x)
        x = self.decoder(lat)
        return x, lat
  
class SeqMLPBlock(nn.Module):
    """
        B, T_IN, C_IN -> B, T_OUT, C_OUT
    """
    def __init__(self, t_in, t_out, c_in, c_out):
        super().__init__()
        self.time_embedding = nn.Linear(t_in, t_out)
        self.channel_embedding = nn.Linear(c_in, c_out)
        
    def forward(self, x):
        # print(x.permute(0, 2, 1).shape)
        x = self.time_embedding(x.permute(0, 2, 1))
        x = F.leaky_relu(x)
        x = self.channel_embedding(x.permute(0, 2, 1))
        x = F.leaky_relu(x)
        
        return x
    
class SeqMLPAutoEncoder(nn.Module):
    def __init__(self, t_in, c_in, lat_size=16):
        super().__init__()
        self.t_in = t_in
        self.c_in = c_in
        self.lat_size = lat_size
        self.pre = SeqMLPBlock(t_in, 128, c_in, 128)
        self.post = SeqMLPBlock(128, t_in, 128, c_in)
        
        self.encoder = nn.Sequential(
            SeqMLPBlock(128, 128, 128, 128),
            SeqMLPBlock(128, 128, 128, 128),
            SeqMLPBlock(128, 128, 128, 128),
            SeqMLPBlock(128, 128, 128, 128),
            SeqMLPBlock(128, 64, 128, 64),
            SeqMLPBlock(64, 32, 64, 32),
            SeqMLPBlock(32, 16, 32, 16),
        )
        self.h2l = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 64),
            nn.Linear(64, lat_size))
        
        self.l2h = nn.Sequential(
            nn.Linear(lat_size, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 256)
        )
        
        self.decoder = nn.Sequential(
            SeqMLPBlock(16, 32, 16, 32),
            SeqMLPBlock(32, 64, 32, 64),
            SeqMLPBlock(64, 128, 64, 128),
            SeqMLPBlock(128, 128, 128, 128),
            SeqMLPBlock(128, 128, 128, 128),
            SeqMLPBlock(128, 128, 128, 128),
            SeqMLPBlock(128, 128, 128, 128)
        )
        

    def forward(self, x):
        N = x.shape[0]
        x = self.pre(x)
        h = self.encoder(x)
        lat = self.h2l(h.reshape(N, -1))
        h = self.l2h(lat).reshape(N, 16, 16)
        x = self.decoder(h)
        x = self.post(x)
        return x, lat


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
