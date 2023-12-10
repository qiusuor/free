from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

class TripletRegDataset(Dataset):
    def __init__(self, data, pos_lowwer_ratio=0.1, pos_upper_ratio=0.1):
        super().__init__()
        self.data = data
        self.pos_lowwer_ratio = pos_lowwer_ratio
        self.pos_upper_ratio = pos_upper_ratio
        self.N = len(self.data)
        self.preprocess()
        
    def preprocess(self):
        self.data= sorted(self.data, key=lambda x: x[1])
        self.labels = [it[1] for it in self.data]
        self.feats = [it[0] for it in self.data]
        self.rank = pd.Series(self.labels).rank(pct=True)
        self.pos_lower = ((self.rank - self.pos_lowwer_ratio) * self.N).astype(int).apply(lambda x: max(0, x)).values
        self.pos_upper = ((self.rank + self.pos_upper_ratio) * self.N).astype(int).apply(lambda x: min(self.N-1, x)).values
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        anchor = self.feats[index].astype(np.float32)
        anchor_label = self.labels[index].astype(np.float32)
        lower_index = self.pos_lower[index]
        upper_index = self.pos_upper[index]
        pos_index = np.random.random_integers(low=lower_index, high=upper_index)
        pos = self.feats[pos_index].astype(np.float32)
        pos_label = self.labels[pos_index].astype(np.float32)
        neg_index = np.random.random_integers(low=upper_index+1, high=self.N+lower_index) % self.N
        neg = self.feats[neg_index].astype(np.float32)
        neg_label = self.labels[neg_index].astype(np.float32)
        
        return anchor, anchor_label, pos, pos_label, neg, neg_label
    
class TripleBinarytDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.N = len(self.data)
        self.preprocess()
        
    def preprocess(self):
        self.labels = [it[1] for it in self.data]
        self.feats = [it[0] for it in self.data]
        self.date = [it[2] for it in self.data]
        self.code_int = [it[3] for it in self.data]
        self.pos_set = [i for i, y in enumerate(self.labels) if y]
        self.neg_set = [i for i, y in enumerate(self.labels) if not y]
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        anchor = self.feats[index].astype(np.float32)
        anchor_label = self.labels[index].astype(np.float32)
        anchor_date = self.date[index]
        anchor_code_int = self.code_int[index]
        pos_set = self.pos_set if anchor_label else self.neg_set
        neg_set = self.neg_set if anchor_label else self.pos_set
        pos_index = np.random.choice(pos_set, 1)[0]
        # print(pos_index)
        pos = self.feats[pos_index].astype(np.float32)
        pos_label = self.labels[pos_index].astype(np.float32)
        pos_date = self.date[pos_index]
        pos_code_int = self.code_int[pos_index]
        neg_index = np.random.choice(neg_set, 1)[0]
        neg = self.feats[neg_index].astype(np.float32)
        neg_label = self.labels[neg_index].astype(np.float32)
        neg_date = self.date[neg_index]
        neg_code_int = self.code_int[neg_index]
        
        return anchor, anchor_label, anchor_date, anchor_code_int, pos, pos_label, pos_date, pos_code_int, neg, neg_label, neg_date, neg_code_int
    