import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_curve, auc, average_precision_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from joblib import load, dump
import _pickle as cPickle
from multiprocessing import Process
from utils import *
from data.fetch_daily import fetch_daily
from data.inject_features import inject_features
from data.inject_labels import inject_labels
from data.inject_embedding import inject_embedding
from data.inject_minute_feature import inject_minute_feature
from matplotlib import pyplot as plt
import shutil
import json
import gc
from data.generate_ltr_data import generate_ltr_data


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
    watches[f"sharp_{k}"] = watches[f"y_next_2_d_high_ratio_topk_{k}_mean"] * watches[f"y_next_2_d_high_ratio_topk_{k}_mean"]
    return miss_cnt, shot_cnt, watches

def pred(gbm, data, groups):
    idx = 0
    preds = []
    epoch = gbm.best_iteration
    for g in groups:
        pred = gbm.predict(data.iloc[idx:idx+g], num_iteration=epoch)
        preds.extend(list(pred))
        idx += g
    return preds
    
def train_lightgbm(argv):
    features, label, train_start_day, train_end_day, val_start_day, val_end_day, n_day, train_len, num_leaves, max_depth, min_data_in_leaf, cache_data, epoch = argv
    params = {
        'task': 'train',  # 执行的任务类型
        'boosting_type': 'gbrt',  # 基学习器
        'objective': 'lambdarank',  # 排序任务(目标函数)
        'metric': 'ndcg',  # 度量的指标(评估函数)
        'metric_freq': 1,  # 每隔多少次输出一次度量结果
        'train_metric': True,  # 训练时就输出度量结果
        'ndcg_at': [10],
        'max_bin': 255,  # 一个整数，表示最大的桶的数量。默认值为 255。lightgbm 会根据它来自动压缩内存。如max_bin=255 时，则lightgbm 将使用uint8 来表示特征的每一个值。
        'num_iterations': 5000,  # 迭代次数，即生成的树的棵数
        'learning_rate': 0.01,  # 学习率
        'feature_fraction': 0.99,
        'bagging_fraction': 0.7,
        'num_leaves': num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "max_depth": max_depth,
        'bagging_freq': 1,
        'verbose': 1,
        "early_stopping_rounds": 20,
        "min_gain_to_split": 0,
        "num_threads": 16,
    }
    
    pred_mode = False
    if epoch > 0:
        params["num_iterations"] = epoch
        params.pop("early_stopping_rounds")
        print(params)
        pred_mode = True
        
    param_des = "_".join([str(train_len), str(num_leaves), str(max_depth), str(min_data_in_leaf)])
    root_dir = EXP_RANK_PRED_DIR if pred_mode else EXP_RANK_DIR
    save_dir = "{}/{}/{}/{}".format(root_dir, label, param_des, to_int_date(val_start_day))
    if os.path.exists(os.path.join(save_dir, "meta.json")):
        return
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    make_dir(save_dir)
    make_dir(cache_data)
    if os.path.exists(cache_data):
        with open(cache_data, 'rb') as f:
            dataset, groups, dates = cPickle.load(f)
    else:
        dataset = []
        groups = []
        dates = []
        for file in os.listdir(DAILY_BY_DATE_DIR):
            if not file.endswith(".pkl"): continue
            path = os.path.join(DAILY_BY_DATE_DIR, file)
            df = joblib.load(path)
            dataset.append(df)
            groups.append(len(df))
            dates.append(int(file[:-4]))
        with open(cache_data, 'wb') as f:
            cPickle.dump((dataset, groups, dates), f)
    train_start_day = to_int_date(train_start_day)
    train_end_day = to_int_date(train_end_day)
    val_start_day = to_int_date(val_start_day)
    val_end_day = to_int_date(val_end_day)
    
    train_dataset, val_dataset = [], []
    train_groups, val_groups = [], []
    for data_day, g, date in zip(dataset, groups, dates):
        if train_start_day <= date <= train_end_day:
            train_dataset.append(data_day)
            train_groups.append(g)
        elif val_start_day <= date <= val_end_day:
            val_dataset.append(data_day)
            val_groups.append(g)
    train_dataset = pd.concat(train_dataset)
    val_dataset = pd.concat(val_dataset)
    del dataset, groups, dates
    gc.collect()
    train_x, train_y = train_dataset[features], train_dataset[label]
    val_x, val_y = val_dataset[features], val_dataset[label]
    lgb_train = lgb.Dataset(train_x, train_y, group=train_groups)
    lgb_eval = lgb.Dataset(val_x, val_y, group=val_groups, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=(lgb_train, lgb_eval),
                    # categorical_feature=["industry"]
                    )

    gbm.save_model(os.path.join(save_dir, "model.txt"))
    joblib.dump(gbm, os.path.join(save_dir, "model.pkl"))
    if epoch <= 0:
        epoch = gbm.best_iteration

    val_y_pred = pred(gbm, val_x, val_groups)
    train_y_pred = pred(gbm, train_x, train_groups)
    train_dataset["pred"] = train_y_pred
    train_dataset.sort_values(by="pred", inplace=True, ascending=False)
    if pred_mode:
        train_dataset[["code", "code_name", "pred", label, "y_next_1d_close_rate", f"y_next_{2}_d_high_ratio", f"y_next_{2}_d_low_ratio", "y_next_{}_d_ret".format(2), "y_next_1d_close_2d_open_rate", "price"]].to_csv(os.path.join(save_dir, "train_set_EPOCH_{}.csv".format(epoch)))


    val_dataset["pred"] = val_y_pred
    res_val = val_dataset[["code", "code_name", "pred", label, "y_next_1d_close_rate", f"y_next_{2}_d_high_ratio", f"y_next_{2}_d_low_ratio", "y_next_{}_d_ret".format(2), "y_next_1d_close_2d_open_rate", "price"]]
    meta = dict()
    meta["config"] = {
        "label": label,
        "train_len": train_len,
        "val_start_day": val_start_day,
        "val_end_day": val_end_day,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_data_in_leaf": min_data_in_leaf,
    }
    meta["info"] = {
        "epoch": epoch,
    }
    meta["daily"] = dict()
    watch_list = [f"y_next_{2}_d_high_ratio", f"y_next_{2}_d_low_ratio", "y_next_1d_close_2d_open_rate", "y_next_1d_close_rate", "y_next_{}_d_ret".format(2)]
    
    labeled_day = 0
    for i, res_i in res_val.groupby("date"):
        res_i.sort_values(by="pred", inplace=True, ascending=False)
        top3_miss, top3_shot, top3_watch = topk_shot(res_i, label, k=3, watch_list=watch_list)
        top5_miss, top5_shot, top5_watch = topk_shot(res_i, label, k=5, watch_list=watch_list)
        top10_miss, top10_shot, top10_watch = topk_shot(res_i, label, k=10, watch_list=watch_list)
        meta["daily"][to_int_date(i)] = {
            "top3_watch": top3_watch,
            "top5_watch": top5_watch,
            "top10_watch": top10_watch,
            "top3_miss": top3_miss,
            "top5_miss": top5_miss,
            "top10_miss": top10_miss,
        }
        if not np.isnan(top3_watch["sharp_3"]):
            labeled_day += 1
        meta["last_val"] = meta["daily"][to_int_date(i)]
        save_file = f"{to_int_date(i)}_T3_{top3_miss}_T5_{top5_miss}_T10_{top10_miss}.csv"
        res_i.to_csv(os.path.join(save_dir, save_file))
    
    def get_topk_watch_templet(which="top3_watch"):
        return {k:0.0 for k in meta["last_val"][which].keys()}
    
    meta["mean_val"] = {
        "top3_watch": get_topk_watch_templet("top3_watch"),
        "top5_watch": get_topk_watch_templet("top5_watch"),
        "top10_watch": get_topk_watch_templet("top10_watch"),
        "top3_miss": 0,
        "top5_miss": 0,
        "top10_miss": 0,
    }
    mean_val = meta["mean_val"]
    # print(meta["daily"])
    
    for daily in meta["daily"].values():
        if np.isnan(daily["top3_watch"]["sharp_3"]): continue
        mean_val["top3_miss"] += daily["top3_miss"] / labeled_day
        mean_val["top5_miss"] += daily["top5_miss"] / labeled_day
        mean_val["top10_miss"] += daily["top10_miss"] / labeled_day
        for watch_key in ["top3_watch", "top5_watch", "top10_watch"]:
            for key in mean_val[watch_key]:
                mean_val[watch_key][key] += daily[watch_key][key] / labeled_day
        
        
    json.dump(meta, open(os.path.join(save_dir, "meta.json"), 'w'), indent=4)
    

def prepare_data(update=False):
    if update:
        fetch_daily()
    remove_dir(EXP_RANK_DIR)
    remove_dir(EXP_RANK_PRED_DIR)
    inject_features()
    # inject_embedding()
    inject_labels()
    generate_ltr_data()
    
    # inject_minute_feature()

def get_n_val_day(label):
    if "y_next_1d_up_to_limit" in label:
        n_day = 1
    elif "y_2_d" in label or "1d_close_2d_open" in label or "y_02" in label or "2d" in label:
        n_day = 2
    elif "y_5_d" in label:
        n_day = 5
    else:
        assert False
    return n_day
