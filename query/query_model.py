from torch import nn
from tcn import TemporalConvNet
import torch.nn.functional as F



class TCN_LSTM(nn.Module):
    def __init__(self, input_size, output_size=1, num_channels=[32, 32, 64, 128, 64, 32, 32], kernel_size=3, dropout=0.0, lstm_layers=4, bidirectional=True):
        super(TCN_LSTM, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.hidden_size = num_channels[-1]
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(num_channels[-1]*2 if bidirectional else num_channels[-1], output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x): # N, L, C
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2) # N, L, C
        output = self.lstm(output)[0]
        output = self.linear(output[:, -1, :])
        # return self.sig(output)
        return output

