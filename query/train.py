import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import sys
import time
import argparse
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.utils import class_weight
import pickle
from config import *
import pandas as pd
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score
from collections import Counter
import platform
import gc
from query_model import TCN_LSTM

feature_cols = ["open", "high", "low", "close", "price", "turn", "volume"]
# feature_cols += get_feature_cols()
label_col = ["y_next_2_d_ret"]
# label_col = ["y_next_2_d_ret_04"]

all_cols = feature_cols + label_col
hist_len = 30
batch_size = 256
epochs = 500
device = torch.device("mps") if platform.machine() == 'arm64' else torch.device("cuda")


def train_val_data_filter(df):
    return df[reserve_n_last(not_limit_line(df).shift(-1)) & (df.isST != 1)]

@hard_disk_cache(force_update=False)
def load_data(n_val_day=30, val_delay_day=30):
    train_data, val_data = [], []
    Xs = []
    for file in tqdm(os.listdir(DAILY_DIR)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        df = joblib.load(path)
        if len(df) < 300: continue
        df = train_val_data_filter(df)
        if len(df) <=0 or df.isST[-1]:
            continue
        if "code_name" not in df.columns or not isinstance(df.code_name[-1], str) or "ST" in df.code_name[-1] or "st" in df.code_name[-1] or "sT" in df.code_name[-1]:
            continue
        # df["date"] = df.index
        # print(df)
        # print(df[label_col].describe())
        df = df[all_cols].iloc[-500:]
        df = df.fillna(0)
        # print(df)
        # df[df.isna()] = 0
        # df = df.loc[(df[label_col[0]] >= 0.79).index]
        df = df.drop(df[df[label_col[0]] == 0].index)
        # print(df)
        # exit(0)
        # print(df[label_col].describe())
        
        Xs.append(df[feature_cols].astype(np.float32))
        for i, data_i in enumerate(list(df.rolling(hist_len))[::-1]):
            feat = data_i[feature_cols].astype(np.float32)
            if (len(feat) < hist_len): continue
            label = data_i[label_col].iloc[-1].astype(np.float32).values
            # print(data_i[label_col], label)
            assert not np.isnan(label), data_i[label_col]
            if i < n_val_day:
                val_data.append([feat.values, label])
            elif i >= n_val_day + val_delay_day:
                train_data.append([feat.values, label])
        # if (len(train_data) > 100000):
        #     break
    #     print(file)
    #     print(label)
    # exit(0)
    Xs = pd.concat(Xs)
    mean = Xs.mean(0).values
    std = Xs.std(0).values
    joblib.dump((mean, std), "query/checkpoint/mean_std.pkl")
    return train_data, val_data



def train():
    global batch_size, epochs
    best_val_loss = 1e9
    model = TCN_LSTM(input_size=len(feature_cols))
    model = model.to(device)
    train_data, val_data = load_data()
    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(val_data, batch_size=batch_size, drop_last=True)


    criterion = nn.L1Loss()
    # criterion = nn.BCELoss()
    binary_cls_task = criterion == nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.0001,
                                betas=[0.9, 0.999],
                                weight_decay = 0.0,
                                amsgrad=False)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.3)
    mean, std = joblib.load("query/checkpoint/mean_std.pkl")
    best_model_path = "query/checkpoint/tcn_lstm.pth"
    for epoch in range(epochs):
        print('EPOCH {} / {}:'.format(epoch + 1, epochs))
        model.train()
        train_loss = []
        train_gt = []
        train_pred = []
        for i, data_i in enumerate(train_loader):
            input, target = data_i
            input = (input - mean) / (std + 1e-9)
            input = torch.FloatTensor(input).to(device)
            target = target.to(device)
            # print(target)
            
            ouput = model(input)
            loss = criterion(ouput, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.data.item())
            train_gt.extend(target.detach().cpu().reshape(-1).numpy().tolist())
            train_pred.extend(ouput.detach().cpu().reshape(-1).numpy().tolist())
            
        scheduler.step()
        train_avg_loss = np.mean(train_loss)
        if binary_cls_task:
            train_ap = average_precision_score(train_gt, train_pred)
            train_auc = roc_auc_score(train_gt, train_pred)
        
        with torch.no_grad():
            model.eval()
            val_loss = []
            val_gt = []
            val_pred = []

            for i, data_i in enumerate(test_loader):
                input, target = data_i
                input = (input - mean) / (std + 1e-9)
                input = torch.FloatTensor(input).to(device)
                ouput = model(input)
                target = target.to(device)
                loss = criterion(ouput, target)
                # print(target, ouput)
                val_loss.append(loss.data.item())
                val_gt.extend(target.cpu().reshape(-1).numpy().tolist())
                val_pred.extend(ouput.cpu().reshape(-1).numpy().tolist())
                
            val_avg_loss = np.mean(val_loss)
            if binary_cls_task:
                val_ap = average_precision_score(val_gt, val_pred)
                val_auc = roc_auc_score(val_gt, val_pred)
            print("Train avg loss: {} Val avg loss: {}".format(train_avg_loss, val_avg_loss))
            if binary_cls_task:
                print("Train AUC: {} Val AUC: {} Train AP: {} Val AP: {}".format(train_auc, val_auc, train_ap, val_ap))
                
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                model_path = best_model_path
                print('    ---> New Best Score: {}. Saving model to {}'.format(best_val_loss, model_path))
                torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    train()
    
