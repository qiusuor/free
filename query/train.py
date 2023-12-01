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
from query.dataset import TripletDataset
from query.tripletLoss import CosineTripletLossWithL1

batch_size = 32
epochs = 5000
max_train_days = 30
n_val_day = 5
val_delay_day = 5
device = torch.device("mps") if platform.machine() == 'arm64' else torch.device("cuda")


def train_val_data_filter(df):
    return df[reserve_n_last(not_limit_line(df).shift(-1)) & (df.isST != 1)]

@hard_disk_cache(force_update=False)
def load_data(n_val_day=n_val_day, val_delay_day=val_delay_day):
    feature_cols = ["open", "high", "low", "close", "price", "turn", "volume"]
    # feature_cols = ["open", "high", "low", "close", "price", "turn", "volume", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "style_feat_shif1_of_y_next_1d_ret_mean_limit_up", "style_feat_shif1_of_y_next_1d_ret_std_limit_up", "style_feat_shif1_of_y_next_1d_ret_mean_limit_up_and_high_price_60", "style_feat_shif1_of_y_next_1d_ret_std_limit_up_and_high_price_60"]
    label_col = ["y_next_2_d_ret"]
    # label_col = ["y_next_2_d_ret_04"]

    all_cols = feature_cols + label_col
    min_hist_len = 30
    max_hist_len = 350
    

    train_data, val_data = [], []
    Xs = []
    whole_data = []
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
        if df.price[-1] > 50: continue
        
        # df["date"] = df.index
        # print(df)
        # print(df[label_col].describe())
        df = df[all_cols].iloc[-500:]
        df = df.fillna(0)
        df[label_col[0]] = df[label_col[0]] - 1
        # df["value"] = df["value"].apply(np.log)
        # print(df)
        # df[df.isna()] = 0
        df = df.iloc[:-2]
        whole_data.append(df)
        Xs.append(df[feature_cols].astype(np.float32))
        # print(len(df))
        for i, data_i in enumerate(list(df.rolling(max_hist_len))[::-1]):
            if i > n_val_day + val_delay_day + max_train_days: break
            feat = data_i[feature_cols].astype(np.float32)
            if (len(feat) < min_hist_len): continue
            feat["mask"] = list(range(1, len(feat)+1))[::-1]
            # print(len(feat))
            
            feat = np.pad(feat.values, pad_width=((max_hist_len-len(feat), 0), (0, 0)), mode="constant", constant_values=0.0)
            # print(feat.shape)
            # print(feat)
            label = data_i[label_col].iloc[-1].astype(np.float32).values
            # print(data_i[label_col], label)
            assert not np.isnan(label), data_i[label_col]
            if i < n_val_day:
                val_data.append([feat, label])
            elif i >= n_val_day + val_delay_day:
                train_data.append([feat, label])
        # if (len(train_data) > 10000):
        #     break
    #     print(file)
    #     print(label)
    # exit(0)
    whole_data = pd.concat(whole_data)
    cPickle.dump(whole_data, open("whole_data.pkl", 'wb'))
    Xs = pd.concat(Xs)
    mean = Xs.mean(0).values
    std = Xs.std(0).values
    joblib.dump((mean, std), "query/checkpoint/mean_std.pkl")
    return train_data, val_data



def train():
    global batch_size, epochs
    best_val_loss = 1e9
    train_data, val_data = load_data()
    train_data_set = TripletDataset(train_data)
    val_data_set = TripletDataset(val_data)
    print(len(train_data))
    
    model = TCN_LSTM(input_size=len(train_data[0][0][0]))
    model = model.to(device)
    
    train_loader = DataLoader(train_data_set, batch_size=batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(val_data_set, batch_size=batch_size, drop_last=True)

    criterion = CosineTripletLossWithL1(margin=0.5, reg_weight=10, triplet_weight=1, device=device)
    
    # criterion = nn.BCELoss()
    binary_cls_task = criterion == nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.0001,
                                betas=[0.9, 0.999],
                                weight_decay = 0.0,
                                amsgrad=False)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000, 3000], gamma=0.3)
    mean, std = joblib.load("query/checkpoint/mean_std.pkl")
    best_model_path = "query/checkpoint/tcn_lstm.pth"
    
    def normalize_core(anchor, anchor_label):
        anchor[:, :, :-1] = (anchor[:, :, :-1] - mean) / (std + 1e-9)
        anchor = torch.FloatTensor(anchor.float()).to(device)
        anchor_label = anchor_label.to(device)
        return [anchor, anchor_label]
    
    def normalize(anchor, anchor_label, pos, pos_label, neg, neg_label):
        return normalize_core(anchor, anchor_label) + normalize_core(pos, pos_label) + normalize_core(neg, neg_label)
    
    for epoch in range(epochs):
        print('EPOCH {} / {}:'.format(epoch + 1, epochs))
        model.train()
        train_loss = []
        train_anchor_loss = []
        train_pos_loss = []
        train_neg_loss = []
        train_triplet_loss = []
        train_gt = []
        train_pred = []
        train_achor_diff_pos = []
        train_achor_diff_neg = []
        
        for i, (anchor, anchor_label, pos, pos_label, neg, neg_label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            anchor, anchor_label, pos, pos_label, neg, neg_label = normalize(anchor, anchor_label, pos, pos_label, neg, neg_label)
            
            anchor_feat, anchor_score, pos_feat, pos_score, neg_feat, neg_score = model(anchor, pos, neg)
            loss, anchor_reg, pos_reg, neg_reg, triplet = criterion(anchor_feat, anchor_score, anchor_label, pos_feat, pos_score, pos_label, neg_feat, neg_score, neg_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.data.item())
            train_anchor_loss.append(anchor_reg.data.item())
            train_pos_loss.append(pos_reg.data.item())
            train_neg_loss.append(neg_reg.data.item())
            train_triplet_loss.append(triplet.data.item())
            train_gt.extend(anchor_score.detach().cpu().reshape(-1).numpy().tolist())
            train_pred.extend(anchor_label.detach().cpu().reshape(-1).numpy().tolist())
            train_achor_diff_pos.append((anchor_label-pos_label).reshape(-1).detach().abs().mean().cpu())
            train_achor_diff_neg.append((anchor_label-neg_label).reshape(-1).detach().abs().mean().cpu())
            
        scheduler.step()
        train_avg_loss = np.mean(train_loss)
        train_avg_anchor_loss = np.mean(train_anchor_loss)
        train_avg_pos_loss = np.mean(train_pos_loss)
        train_avg_neg_loss = np.mean(train_neg_loss)
        train_avg_triplet_loss = np.mean(train_triplet_loss)
        train_avg_anchor_diff_pos = np.mean(train_achor_diff_pos)
        train_avg_anchor_diff_neg = np.mean(train_achor_diff_neg)
        if binary_cls_task:
            train_ap = average_precision_score(train_gt, train_pred)
            train_auc = roc_auc_score(train_gt, train_pred)
        
        with torch.no_grad():
            model.eval()
            val_loss = []
            val_anchor_loss = []
            val_pos_loss = []
            val_neg_loss = []
            val_triplet_loss = []
            val_gt = []
            val_pred = []
            

            for i, (anchor, anchor_label, pos, pos_label, neg, neg_label) in enumerate(test_loader):
                anchor, anchor_label, pos, pos_label, neg, neg_label = normalize(anchor, anchor_label, pos, pos_label, neg, neg_label)
                
                anchor_feat, anchor_score, pos_feat, pos_score, neg_feat, neg_score = model(anchor, pos, neg)
                loss, anchor_reg, pos_reg, neg_reg, triplet = criterion(anchor_feat, anchor_score, anchor_label, pos_feat, pos_score, pos_label, neg_feat, neg_score, neg_label)
                val_loss.append(loss.data.item())
                val_anchor_loss.append(anchor_reg.data.item())
                val_pos_loss.append(pos_reg.data.item())
                val_neg_loss.append(neg_reg.data.item())
                val_triplet_loss.append(triplet.data.item())
                val_gt.extend(anchor_score.detach().cpu().reshape(-1).numpy().tolist())
                val_pred.extend(anchor_label.detach().cpu().reshape(-1).numpy().tolist())
                
            val_avg_loss = np.mean(val_loss)
            val_avg_anchor_loss = np.mean(val_anchor_loss)
            val_avg_pos_loss = np.mean(val_pos_loss)
            val_avg_neg_loss = np.mean(val_neg_loss)
            val_avg_triplet_loss = np.mean(val_triplet_loss)
            if binary_cls_task:
                val_ap = average_precision_score(val_gt, val_pred)
                val_auc = roc_auc_score(val_gt, val_pred)
            print("anchor-pos: {} anchor-neg: {} Train avg loss: {} Val avg loss: {} Train anchor: {} Val anchor: {} Train pos: {} Val pos: {} Train neg: {} Val neg: {} Train triplet: {} Val triplet: {}".format(train_avg_anchor_diff_pos, train_avg_anchor_diff_neg, train_avg_loss, val_avg_loss, train_avg_anchor_loss, val_avg_anchor_loss, train_avg_pos_loss, val_avg_pos_loss, train_avg_neg_loss, val_avg_neg_loss, train_avg_triplet_loss, val_avg_triplet_loss))
            if binary_cls_task:
                print("Train AUC: {} Val AUC: {} Train AP: {} Val AP: {}".format(train_auc, val_auc, train_ap, val_ap))
                
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                model_path = best_model_path
                print('    ---> New Best Score: {}. Saving model to {}'.format(best_val_loss, model_path))
                torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    train()
    
