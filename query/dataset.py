from torch.utils.data import Dataset
import torch
import numpy as np

class TripletDataset(Dataset):
    def __init__(self, df, feature_cols, label_col, pos_lowwer_ratio, pos_upper_ratio):
        super().__init__()
        self.df = df
        self.pos_lowwer_ratio = pos_lowwer_ratio
        self.pos_upper_ratio = pos_upper_ratio
        self.feature_cols =feature_cols
        self.label_col =label_col
        self.N = len(self.df)
        
    def preprocess(self):
        self.df= self.df.sort_values(by=self.label_col)
        self.df["rank"] = self.df[self.label_col].rank(pct=True)
        self.df["pos_lower"] = ((self.self.df["rank"] - self.pos_lowwer_ratio) * self.N).astype(int).apply(lambda x: max(0, x))
        self.df["pos_upper"] = ((self.self.df["rank"] + self.pos_upper_ratio) * self.N).astype(int).apply(lambda x: min(self.N, x))
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        anchor = self.df.iloc[index][self.feature_cols].astype(np.float32)
        label = self.df.iloc[index][self.label_col].astype(np.float32)
        lower_index = self.df.iloc[index]["pos_lower"]
        upper_index = self.df.iloc[index]["pos_upper"]
        pos_index = np.random.random_integers(low=lower_index, high=upper_index)
        pos = self.df.iloc[pos_index][self.feature_cols].astype(np.float32)
        pos_label = self.df.iloc[pos_index][self.label_col].astype(np.float32)
        neg_index = np.random.random_integers(low=upper_index, high=self.N+lower_index) % self.N
        neg = self.df.iloc[neg_index][self.feature_cols].astype(np.float32)
        neg_label = self.df.iloc[neg_index][self.label_col].astype(np.float32)
        
        return anchor, label, pos, pos_label, neg, neg_label
    