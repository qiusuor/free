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
from sklearn.metrics import mean_squared_error, roc_curve, auc, average_precision_score, roc_auc_score
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
from query.dataset import TripleBinarytDataset
from query.tripletLoss import TripletWrapLoss
import json

batch_size = 512
epochs = 5000
max_train_days = 90
n_val_day = 5
val_delay_day = 10
device = torch.device("mps") if platform.machine() == 'arm64' else torch.device("cuda")
label_field = "y_next_2_d_ret_04"
save_dir = "query/exp"

trade_days = get_trade_days(update=False)
trunc_index = bisect.bisect_right(trade_days, SEARCH_END_DAY)
trade_days = trade_days[:trunc_index]

val_start_day = trade_days[-n_val_day]
val_end_day = trade_days[-1]
train_start_day = trade_days[-n_val_day-val_delay_day-max_train_days]
train_end_day = trade_days[-n_val_day-val_delay_day]

def topk_shot(data, label, k=10, watch_list=[]):
    gt_labels = data[label].values[:k]
    shot_cnt = 0
    miss_cnt = 0
    for label in gt_labels:
        if label:
            shot_cnt += 1
        else:
            miss_cnt += 1
    watches = dict()
    for watch in watch_list:
        watches[watch+f"_topk_{k}_max"] = data[watch][:k].max()
        watches[watch+f"_topk_{k}_min"] = data[watch][:k].min()
        watches[watch+f"_topk_{k}_mean"] = data[watch][:k].mean()
    watches[f"sharp_{k}"] = (watches[f"y_next_2_d_high_ratio_topk_{k}_mean"] + watches[f"y_next_2_d_low_ratio_topk_{k}_mean"]) / 2
    return miss_cnt, shot_cnt, watches

def train_val_data_filter(df):
    return df[reserve_n_last(not_limit_line(df).shift(-1)) & (df.isST != 1)]

@hard_disk_cache(force_update=False)
def load_data():
    # feature_cols = get_feature_cols()
    feature_cols = ["open", "high", "low", "close", "price", "turn", "volume", "value"]
    label_col = [label_field]

    all_cols = feature_cols + label_col
    min_hist_len = 250
    max_hist_len = 250
    

    train_data, val_data = [], []
    Xs = []
    all_val_data = []
    all_val_data_cols = []
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
        
        df = df.iloc[-500:]
        df["date_int"] = list(map(lambda x: to_int_date(x), df.index))
        df["code_int"] = df["code"].apply(lambda x: int(x[-6:]))
        df = df.fillna(0)
        df = df.iloc[:-2, :]
        # df = df.sort_index()
        Xs.append(df[feature_cols].astype(np.float32))
        all_val_data_cols = df.columns
        for i, data_i in enumerate(list(df.rolling(max_hist_len))[::-1]):
            feat = data_i[feature_cols].astype(np.float32)
            if (len(feat) < min_hist_len): break
            # if i > n_val_day + val_delay_day + max_train_days: break
            feat["mask"] = list(range(1, len(feat)+1))[::-1]
            feat = np.pad(feat.values, pad_width=((max_hist_len-len(feat), 0), (0, 0)), mode="constant", constant_values=0.0)
            label = data_i[label_col].iloc[-1].astype(np.float32).values
            end_date = data_i.date_int[-1]
            code_int = data_i.code_int[-1]
            assert not np.isnan(label), data_i[label_col]
            if val_end_day >= end_date >= val_start_day:
                val_data.append([feat, label, end_date, code_int])
                all_val_data.append(data_i.iloc[-1].values)  
            elif train_start_day <= end_date <= train_end_day:
                train_data.append([feat, label, end_date, code_int])
    Xs = pd.concat(Xs)
    mean = Xs.mean(0).values
    std = Xs.std(0).values
    all_val_data = pd.DataFrame(all_val_data, columns=all_val_data_cols)
    with open("all_val_dataset.pkl", "wb") as f:
        cPickle.dump(all_val_data, f)
    joblib.dump((mean, std), "query/checkpoint/mean_std_cls.pkl")
    return train_data, val_data



def train():
    global batch_size, epochs
    best_val_ap = -1e9
    train_data, val_data = load_data()
    train_data_set = TripleBinarytDataset(train_data)
    val_data_set = TripleBinarytDataset(val_data)
    with open("all_val_dataset.pkl", "rb") as f:
        all_val_data = cPickle.load(f)
        # print(all_val_data[all_val_data.code_int == 670])
        # exit(0)
    print("Tota train sample:", len(train_data))
    print("Tota Val sample:", len(val_data_set))
    
    model = TCN_LSTM(input_size=len(train_data[0][0][0]))
    model = model.to(device)
    
    train_loader = DataLoader(train_data_set, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=64)
    test_loader = DataLoader(val_data_set, batch_size=batch_size, drop_last=False, num_workers=64)

    criterion = TripletWrapLoss(margin=0.5, loss=nn.BCELoss(), loss_weight=1, triplet_weight=1, device=device)
    
    binary_cls_task = True
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.0001,
                                betas=[0.9, 0.999],
                                weight_decay = 0.0,
                                amsgrad=False)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000, 3000], gamma=0.3)
    mean, std = joblib.load("query/checkpoint/mean_std_cls.pkl")
    best_model_path = "query/checkpoint/tcn_lstm_cls.pth"
    
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
        
        for i, (anchor, anchor_label, anchor_date, anchor_code_int, pos, pos_label, pos_date, pos_code_int, neg, neg_label, neg_date, neg_code_int) in tqdm(enumerate(train_loader), total=len(train_loader)):
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
            train_gt.extend(anchor_label.detach().cpu().reshape(-1).numpy().tolist())
            train_pred.extend(anchor_score.detach().cpu().reshape(-1).numpy().tolist())
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
            val_pred_data = []
            

            for i, (anchor, anchor_label, anchor_date, anchor_code_int, pos, pos_label, pos_date, pos_code_int, neg, neg_label, neg_date, neg_code_int) in enumerate(test_loader):
                anchor, anchor_label, pos, pos_label, neg, neg_label = normalize(anchor, anchor_label, pos, pos_label, neg, neg_label)
                
                anchor_feat, anchor_score, pos_feat, pos_score, neg_feat, neg_score = model(anchor, pos, neg)
                loss, anchor_reg, pos_reg, neg_reg, triplet = criterion(anchor_feat, anchor_score, anchor_label, pos_feat, pos_score, pos_label, neg_feat, neg_score, neg_label)
                val_loss.append(loss.data.item())
                val_anchor_loss.append(anchor_reg.data.item())
                val_pos_loss.append(pos_reg.data.item())
                val_neg_loss.append(neg_reg.data.item())
                val_triplet_loss.append(triplet.data.item())
                val_gt.extend(anchor_label.detach().cpu().reshape(-1).numpy().tolist())
                val_pred.extend(anchor_score.detach().cpu().reshape(-1).numpy().tolist())
                val_pred_data.extend(list(zip(anchor_score.detach().cpu().reshape(-1).numpy().tolist(), anchor_date.detach().cpu().reshape(-1).numpy().tolist(), anchor_code_int.detach().cpu().reshape(-1).numpy().tolist())))
                
            val_pred_data = pd.DataFrame(val_pred_data, columns=["pred", "date_int", "code_int"])
            val_pred_data["date_int"] = val_pred_data["date_int"].astype(int)
            val_pred_data["code_int"] = val_pred_data["code_int"].astype(int)
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
                
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                model_path = best_model_path
                print('    ---> New Best Score: {}. Saving model to {}'.format(best_val_ap, model_path))
                torch.save(model.state_dict(), model_path)
                # for b in val_pred_data.groupby(by=["date_int", "code_int"]):
                #     assert len(b[1]) == 1
                # for b in all_val_data.groupby(by=["date_int", "code_int"]):
                #     assert len(b[1]) == 1
                    # print(b)
                # print(val_pred_data.dtypes)
                # print(all_val_data.dtypes)
                os.system("rm -rf {}/*".format(save_dir))
                all_val_data_df = pd.merge(all_val_data, val_pred_data, on=["date_int", "code_int"], how="left")
                # print(all_val_data)
                res_val = all_val_data_df[["date_int", "code", "code_name", "pred", label_field, "y_next_1d_close_rate", f"y_next_{2}_d_high_ratio", f"y_next_{2}_d_low_ratio", "y_next_{}_d_ret".format(2), "y_next_1d_close_2d_open_rate", "price"]]
                meta = dict()
                # meta["config"] = {
                #     "label": label,
                #     "train_len": train_len,
                #     "val_start_day": to_int_date(val_start_day),
                #     "val_end_day": to_int_date(val_end_day),
                #     "num_leaves": num_leaves,
                #     "max_depth": max_depth,
                #     "min_data_in_leaf": min_data_in_leaf,
                # }
                meta["info"] = {
                    "epoch": epoch,
                    "train_auc": train_auc,
                    "train_ap": train_ap,
                    "val_auc:": val_auc,
                    "val_ap": val_ap,
                }
                meta["daily"] = dict()
                watch_list = [f"y_next_{2}_d_high_ratio", f"y_next_{2}_d_low_ratio", "y_next_1d_close_2d_open_rate", "y_next_1d_close_rate", "y_next_{}_d_ret".format(2)]
                
                labeled_day = 0
                for i, res_i in res_val.groupby("date_int"):
                    res_i.sort_values(by="pred", inplace=True, ascending=False)
                    top3_miss, top3_shot, top3_watch = topk_shot(res_i, label_field, k=3, watch_list=watch_list)
                    top5_miss, top5_shot, top5_watch = topk_shot(res_i, label_field, k=5, watch_list=watch_list)
                    top10_miss, top10_shot, top10_watch = topk_shot(res_i, label_field, k=10, watch_list=watch_list)
                    fpr, tpr, thresh = roc_curve(res_i[label_field], res_i.pred)
                    auc_score = auc(fpr, tpr)
                    ap = average_precision_score(res_i[label_field], res_i.pred)
                    meta["daily"][i] = {
                        "auc": auc_score,
                        "ap": ap,
                        "top3_watch": top3_watch,
                        "top5_watch": top5_watch,
                        "top10_watch": top10_watch,
                        "top3_miss": top3_miss,
                        "top5_miss": top5_miss,
                        "top10_miss": top10_miss,
                    }
                    if not np.isnan(top3_watch["sharp_3"]):
                        labeled_day += 1
                    meta["last_val"] = meta["daily"][i]
                    save_file = f"{i}_T3_{top3_miss}_T5_{top5_miss}_T10_{top10_miss}_AP_{ap}_AUC_{auc_score}.csv"
                    res_i.to_csv(os.path.join(save_dir, save_file), index=False)
                
                def get_topk_watch_templet(which="top3_watch"):
                    return {k:0.0 for k in meta["last_val"][which].keys()}
                
                meta["mean_val"] = {
                    "auc": 0,
                    "ap": 0,
                    "top3_watch": get_topk_watch_templet("top3_watch"),
                    "top5_watch": get_topk_watch_templet("top5_watch"),
                    "top10_watch": get_topk_watch_templet("top10_watch"),
                    "top3_miss": 0,
                    "top5_miss": 0,
                    "top10_miss": 0,
                }
                mean_val = meta["mean_val"]
                
                for daily in meta["daily"].values():
                    if np.isnan(daily["auc"]): continue
                    mean_val["auc"] += daily["auc"] / labeled_day
                    mean_val["ap"] += daily["ap"] / labeled_day
                    mean_val["top3_miss"] += daily["top3_miss"] / labeled_day
                    mean_val["top5_miss"] += daily["top5_miss"] / labeled_day
                    mean_val["top10_miss"] += daily["top10_miss"] / labeled_day
                    for watch_key in ["top3_watch", "top5_watch", "top10_watch"]:
                        for key in mean_val[watch_key]:
                            mean_val[watch_key][key] += daily[watch_key][key] / labeled_day
                    
                    
                json.dump(meta, open(os.path.join(save_dir, "meta.json"), 'w'), indent=4)

if __name__ == "__main__":
    train()
    
