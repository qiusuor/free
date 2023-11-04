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


warnings.filterwarnings("ignore")

batch_size = 128
train_val_split = 0.7
features = get_feature_cols()
label = "y_next_1d_up_to_limit"

def generate_nn_train_val():
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    for file in tqdm(os.listdir(DAILY_DIR)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        df = joblib.load(path)
        feat, label = df[features].values, df[label].valuse
        X,Y = (train_x, train_y) if np.random.random() < train_val_split else (val_x, val_y)

        X.append(feat)
        Y.append(label)
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    val_x = np.concatenate(val_x)
    val_y = np.concatenate(val_y)

        
    
def train():      
    x_train, y_train, x_test, y_test = generate_nn_train_val()
    num_outputs = 3
    class_weights=class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_train), y=y_train)
    print("class weight:", list(class_weights))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float, device=device))
    # print(x_train.shape)
    x_train = DataLoader(x_train, batch_size=batch_size)
    y_train = DataLoader(y_train, batch_size=batch_size)
    x_test = DataLoader(x_test, batch_size=batch_size)
    y_test = DataLoader(y_test, batch_size=batch_size)   
                
                
        
        
   