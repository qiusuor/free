import numpy as np
import pandas as pd
from talib import abstract
import talib
from multiprocessing import Pool
from config import *
from utils import *
from tqdm import tqdm
from joblib import dump
import warnings
from sklearn.utils import class_weight
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import platform


warnings.filterwarnings("ignore")

batch_size = 128
train_val_split = 0.7

features = get_feature_cols()
label = "y_02_107"

def generate_nn_train_val():
    train_data, val_data = [], []
    for file in tqdm(os.listdir(DAILY_DIR)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        df = joblib.load(path)
        data = train_data if np.random.random() < train_val_split else val_data
        data.append(df.iloc[-300:])
    train_data = pd.concat(train_data)
    val_data = pd.concat(val_data)
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    val_data = val_data.sample(frac=1).reset_index(drop=True)
    
    return train_data, val_data
    

        
    
def train():
    best_ap = 0
    best_model_path = "rank/models/checkpoint/mlp_ranker.pth"
    torch.set_default_dtype(torch.float32)
    device = torch.device("mps") if platform.machine() == 'arm64' else torch.device("cuda")
    print("training on device: ", device)
    
    train_data, val_data = generate_nn_train_val()
    train_data = train_data[get_feature_cols() + [label]].values
    val_data = val_data[get_feature_cols() + [label]].values
    
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    for data in train_loader:
        x, y = data[:,:-1], data[:,-1]
        print(x.shape)
        exit(0)
                
        
        
if __name__ == "__main__":
    train()
    