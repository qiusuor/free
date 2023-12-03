from torch import nn
from tcn import TemporalConvNet
import torch.nn.functional as F



class MlpQuery(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(MlpQuery, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.score = nn.Linear(128, output_size)
        
    def single_embedding_flow(self, x):
        feat = self.mlp(x)
        score = self.score(feat)
        return feat, score

    def forward(self, anchor, pos, neg): # N, L, C
        anchor_feat, anchor_score = self.single_embedding_flow(anchor)
        pos_feat, pos_score = self.single_embedding_flow(pos)
        neg_feat, neg_score = self.single_embedding_flow(neg)
        return anchor_feat, anchor_score, pos_feat, pos_score, neg_feat, neg_score

