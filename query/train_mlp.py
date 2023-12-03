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
from mlp_query_model import MlpQuery
from query.dataset import TripletDataset
from query.tripletLoss import CosineTripletLossWithL1

batch_size = 512
epochs = 5000
# max_train_days = 30
# n_val_day = 5
# val_delay_day = 5
device = torch.device("mps") if platform.machine() == 'arm64' else torch.device("cuda")


def train_val_data_filter(df):
    return df[reserve_n_last(not_limit_line(df).shift(-1)) & (df.isST != 1)]


@hard_disk_cache(force_update=False)
def load_data():
    feature_cols = get_feature_cols()
    # feature_cols = ["open", "high", "low", "close", "price", "turn", "volume", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "style_feat_shif1_of_y_next_1d_ret_mean_limit_up", "style_feat_shif1_of_y_next_1d_ret_std_limit_up", "style_feat_shif1_of_y_next_1d_ret_mean_limit_up_and_high_price_60", "style_feat_shif1_of_y_next_1d_ret_std_limit_up_and_high_price_60"]
    label_col = ["y_next_2_d_ret"]
    # label_col = ["y_next_2_d_ret_04"]
    dataset = []
    


    dataset = []
    for file in os.listdir(DAILY_DIR):
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
        df["date"] = df.index
        df = df.iloc[-350:]
        dataset.append(df)
    dataset = pd.concat(dataset, axis=0)
    dataset = dataset.fillna(0)
    Xs = dataset[feature_cols]
    mean = Xs.mean(0).values
    std = Xs.std(0).values
    joblib.dump((mean, std), "query/checkpoint/mean_std_mlp.pkl")
    
    trade_days = get_trade_days(update=False)
    trunc_index = bisect.bisect_right(trade_days, SEARCH_END_DAY)
    trade_days = trade_days[:trunc_index]
    train_start_day = to_date(trade_days[-250])
    train_end_day = to_date(trade_days[-50])
    val_start_day = to_date(trade_days[-30])
    val_end_day = to_date(trade_days[-1])
    train_dataset = dataset[(dataset.date >= train_start_day) & (dataset.date <= train_end_day)]
    # print(len(train_dataset))
    # exit(0)
    val_dataset = dataset[(dataset.date >= val_start_day) & (dataset.date <= val_end_day)]
    train_data = list(zip(train_dataset[feature_cols].values, train_dataset[label_col].values))
    # print(len(train_data))
    val_data = list(zip(val_dataset[feature_cols].values, val_dataset[label_col].values))
    # print(len(val_data))
    return train_data, val_data



def train():
    global batch_size, epochs
    best_val_loss = 1e9
    train_data, val_data = load_data()
    train_data_set = TripletDataset(train_data)
    val_data_set = TripletDataset(val_data)
    print(len(train_data))
    
    model = MlpQuery(input_size=len(train_data[0][0]), output_size=1)
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
    mean, std = joblib.load("query/checkpoint/mean_std_mlp.pkl")
    best_model_path = "query/checkpoint/query_mlp.pth"
    
    def normalize_core(anchor, anchor_label):
        anchor = (anchor - mean) / (std + 1e-9)
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
            
            # if i > 100: break
            
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
                # print(anchor_feat[0])
                loss, anchor_reg, pos_reg, neg_reg, triplet = criterion(anchor_feat, anchor_score, anchor_label, pos_feat, pos_score, pos_label, neg_feat, neg_score, neg_label)
                # print(anchor_reg)
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
    
