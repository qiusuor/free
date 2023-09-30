import torch
import pickle
import pandas as pd
import numpy as np
import tensorboard as tb
import torch.nn as nn
from torch.nn import LSTMCell
import torch.nn.functional as F



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
    def __init__(self, input_size, lat_size, hidden_size=16):
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
            nn.Linear(hidden_size, lat_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(lat_size, hidden_size),
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
            SeqMLPBlock(128, 64, 128, 64),
            SeqMLPBlock(64, 32, 64, 32),
            SeqMLPBlock(32, 16, 32, 16),
            SeqMLPBlock(16, 8, 16, 8),
        )
        self.h2l = nn.Linear(64, lat_size)
        self.l2h = nn.Linear(lat_size, 64)
        self.decoder = nn.Sequential(
            SeqMLPBlock(8, 16, 8, 16),
            SeqMLPBlock(16, 32, 16, 32),
            SeqMLPBlock(32, 64, 32, 64),
            SeqMLPBlock(64, 128, 64, 128)
        )
        

    def forward(self, x):
        N = x.shape[0]
        x = self.pre(x)
        h = self.encoder(x)
        lat = self.h2l(h.reshape(N, -1))
        h = self.l2h(lat).reshape(N, 8, 8)
        x = self.decoder(h)
        x = self.post(x)
        return x, lat
    

class TransformerAutoEncoder(nn.Module):
    def __init__(self, input_size, output_size, lat_size, hidden_size=128, num_layers=3, dropout=0, batch_first=True, bidirectional=True, n_class=3):
        raise NotImplementedError

    def forward(self, seq):
        raise NotImplementedError
    


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
