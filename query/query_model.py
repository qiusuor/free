from torch import nn
from tcn import TemporalConvNet
import torch.nn.functional as F



class TCN_LSTM(nn.Module):
    def __init__(self, input_size, output_size=1, num_channels=[32, 32, 64, 128, 64, 32, 32], kernel_size=3, dropout=0.0, lstm_layers=1):
        super(TCN_LSTM, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.hidden_size = num_channels[-1]
        self.lstm = nn.LSTM(self.hidden_size, hidden_size=self.hidden_size, num_layers=lstm_layers, batch_first=True)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sigmoid = nn.Sigmoid()
        
    def single_embedding_flow(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2) # N, L, C
        feat = self.lstm(output)[0][:, -1, :]
        score = self.linear(feat)
        return feat, self.sigmoid(score)

    def forward(self, anchor, pos, neg): # N, L, C
        anchor_feat, anchor_score = self.single_embedding_flow(anchor)
        pos_feat, pos_score = self.single_embedding_flow(pos)
        neg_feat, neg_score = self.single_embedding_flow(neg)
        return anchor_feat, anchor_score, pos_feat, pos_score, neg_feat, neg_score

