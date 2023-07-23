import torch
import pickle
import pandas as pd
import numpy as np
import tensorboard as tb
import torch.nn as nn
from torch.nn import LSTM



class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size, output_size, lat_size, hidden_size=128, num_layers=3, dropout=0, batch_first=True, bidirectional=True, n_class=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.D = 2 if self.bidirectional == True else 1
        self.num_layers = num_layers
        self.pre = nn.Linear(input_size, hidden_size)
        self.encoder = LSTM(hidden_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        self.e2l = nn.Sequential(
            nn.Linear(self.D*self.num_layers*2*hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, lat_size),
            nn.LeakyReLU(),
        )
        self.l2c =  nn.Sequential(
            nn.Linear(lat_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, self.D*self.num_layers*2*hidden_size),
            nn.LeakyReLU(),
        )
        self.decoder = LSTM(hidden_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        self.post = nn.Linear(self.D*hidden_size, output_size)
        
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
        N = seq.size(0)
        # print(seq.shape)
        o = self.pre(seq)
        _, (h, c) = self.encoder(o)
        # lat = self.e2l(torch.cat([h, c], dim=0).reshape(N, -1))
        # o = self.l2c(lat)
        o = torch.cat([h, c], dim=0).reshape(N, -1)
        o, (h, c) = self.decoder(torch.zeros(seq.size(0), seq.size(1), self.hidden_size).to(device), (o[:,:self.D*self.num_layers*self.hidden_size].reshape(-1, N, self.hidden_size), o[:,self.D*self.num_layers*self.hidden_size:].reshape(-1, N, self.hidden_size)))
        o = self.post(o)
        # y = self.classifier(lat)
        
        return o, o
    

class TransformerAutoEncoder(nn.Module):
    def __init__(self, input_size, output_size, lat_size, hidden_size=128, num_layers=3, dropout=0, batch_first=True, bidirectional=True, n_class=3):
        raise NotImplementedError
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.D = 2 if self.bidirectional == True else 1
        self.num_layers = num_layers
        self.pre = nn.Linear(input_size, hidden_size)
        self.encoder = LSTM(hidden_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        self.e2l = nn.Sequential(
            nn.Linear(self.D*self.num_layers*2*hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, lat_size),
            nn.LeakyReLU(),
        )
        self.l2c =  nn.Sequential(
            nn.Linear(lat_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, self.D*self.num_layers*2*hidden_size),
            nn.LeakyReLU(),
        )
        self.decoder = LSTM(hidden_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        self.post = nn.Linear(self.D*hidden_size, output_size)
        self.classifier = nn.Sequential(
            nn.Linear(lat_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, n_class),
        )
        

    def forward(self, seq):
        raise NotImplementedError
        N = seq.size(0)
        # print(seq.shape)
        o = self.pre(seq)
        _, (h, c) = self.encoder(o)
        lat = self.e2l(torch.cat([h, c], dim=0).reshape(N, -1))
        o = self.l2c(lat)
        o, (h, c) = self.decoder(torch.zeros(seq.size(0), seq.size(1), self.hidden_size).cuda(), (o[:,:self.D*self.num_layers*self.hidden_size].reshape(-1, N, self.hidden_size), o[:,self.D*self.num_layers*self.hidden_size:].reshape(-1, N, self.hidden_size)))
        o = self.post(o)
        y = self.classifier(lat)
        
        return o, y, lat
        
    


