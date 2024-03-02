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
from data.inject_embedding import inject_embedding
from data.inject_minute_feature import inject_minute_feature
from data.generate_style_leaning_feature import generate_style_learning_info
from matplotlib import pyplot as plt
import shutil
import json
import gc
from data.generate_ltr_data import generate_ltr_data
from data.inject_labels import inject_labels


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
    watches[f"sharp_{k}"] = (watches[f"y_next_2_d_high_ratio_topk_{k}_mean"] + watches[f"y_next_2_d_high_ratio_topk_{k}_mean"]) / 2
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
    
def style_filter(train_set, train_gps, train_dates, train_start_day, train_end_day, pred_set):
    train_val_split = 0.6
    skip_days = 10
    filed = "style_feat_y_open_close_mean_limit_up_1d"
    style_feat = joblib.load(STYLE_FEATS)[to_date(train_start_day):to_date(train_end_day)]
    style_feat["day"] = style_feat.index
    thresh = list(style_feat[filed].quantile(np.linspace(0, 1, 5)).values)
    mean_val = pred_set[filed].mean()
    cut_index = bisect.bisect_left(thresh, mean_val)
    if cut_index >= len(thresh)-1:
        cut_index = len(thresh)-2
    
    left_bound, right_bound = thresh[cut_index], thresh[cut_index+1]
    
    filtered_date = list(sorted(style_feat[(style_feat[filed] >= -1e9) & (style_feat[filed] <= 1e9)]["day"].apply(to_int_date).values))
    N = len(filtered_date) - skip_days
    train_days = set(filtered_date[:int(N*train_val_split)])
    val_days = set(filtered_date[-int(N-N*train_val_split):])
    splited_train_set, splited_train_gps = [], []
    splited_val_set, splited_val_gps = [], []
    print(sorted(train_days))
    print(sorted(val_days))
    for data, gp, day in zip(train_set, train_gps, train_dates):
        # print(day)
        if day in train_days:
            splited_train_set.append(data)
            splited_train_gps.append(gp)
        elif day in val_days:
            splited_val_set.append(data)
            splited_val_gps.append(gp)
    splited_train_set = pd.concat(splited_train_set)
    splited_val_set = pd.concat(splited_val_set)
    
    return splited_train_set, splited_train_gps, splited_val_set, splited_val_gps

def train_lightgbm(argv):
    features, label, train_start_day, train_end_day, pred_start_day, pred_end_day, n_day, train_len, num_leaves, max_depth, min_data_in_leaf, cache_data, epoch = argv
    params = {
        'task': 'train',  # 执行的任务类型
        'boosting_type': 'gbrt',  # 基学习器
        'objective': 'lambdarank',  # 排序任务(目标函数)
        'metric': 'ndcg',  # 度量的指标(评估函数)
        'metric_freq': 1,  # 每隔多少次输出一次度量结果
        'train_metric': True,  # 训练时就输出度量结果
        'ndcg_at': [3, 5, 10, 30, 50, 100, 200],
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
        "max_position": 100,
    }
    
    pred_mode = False
    if epoch > 0:
        params["num_iterations"] = epoch
        params.pop("early_stopping_rounds")
        print(params)
        pred_mode = True
        
    param_des = "_".join([str(train_len), str(num_leaves), str(max_depth), str(min_data_in_leaf)])
    root_dir = EXP_RANK_PRED_DIR if pred_mode else EXP_RANK_DIR
    save_dir = "{}/{}/{}/{}".format(root_dir, label, param_des, to_int_date(pred_start_day))
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
            df = df[df["limit_up_1d"]]
            dataset.append(df)
            groups.append(len(df))
            dates.append(int(file[:-4]))
        with open(cache_data, 'wb') as f:
            cPickle.dump((dataset, groups, dates), f)
    train_start_day = to_int_date(train_start_day)
    train_end_day = to_int_date(train_end_day)
    pred_start_day = to_int_date(pred_start_day)
    pred_end_day = to_int_date(pred_end_day)
    
    train_dataset, pred_dataset = [], []
    train_groups, pred_groups = [], []
    train_dates = []
    for data_day, g, date in zip(dataset, groups, dates):
        if train_start_day <= date <= train_end_day:
            train_dataset.append(data_day)
            train_groups.append(g)
            train_dates.append(date)
        elif pred_start_day <= date <= pred_end_day:
            pred_dataset.append(data_day)
            pred_groups.append(g)
    pred_dataset = pd.concat(pred_dataset)
    splited_train_set, splited_train_gps, splited_val_set, splited_val_gps = style_filter(train_dataset, train_groups, train_dates, train_start_day, train_end_day, pred_dataset)
    del dataset, groups, dates
    gc.collect()
    train_x, train_y = splited_train_set[features], splited_train_set[label]
    val_x, val_y = splited_val_set[features], splited_val_set[label]
    pred_x, pred_y = pred_dataset[features], pred_dataset[label]
    
    lgb_train = lgb.Dataset(train_x, train_y, group=splited_train_gps)
    lgb_val = lgb.Dataset(val_x, val_y, group=splited_val_gps, reference=lgb_train)
    lgb_pred = lgb.Dataset(pred_x, pred_y, group=pred_groups, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=(lgb_train, lgb_val),
                    # categorical_feature=["industry"]
                    )
    train_ndcg_3 = gbm.best_score['training']['ndcg@3']
    train_ndcg_5 = gbm.best_score['training']['ndcg@5']
    train_ndcg_10 = gbm.best_score['training']['ndcg@10']
    train_ndcg_30 = gbm.best_score['training']['ndcg@30']
    train_ndcg_50 = gbm.best_score['training']['ndcg@50']
    train_ndcg_100 = gbm.best_score['training']['ndcg@100']
    train_ndcg_200 = gbm.best_score['training']['ndcg@200']
    
    val_ndcg_3 = gbm.best_score['valid_1']['ndcg@3']
    val_ndcg_5 = gbm.best_score['valid_1']['ndcg@5']
    val_ndcg_10 = gbm.best_score['valid_1']['ndcg@10']
    val_ndcg_30 = gbm.best_score['valid_1']['ndcg@30']
    val_ndcg_50 = gbm.best_score['valid_1']['ndcg@50']
    val_ndcg_100 = gbm.best_score['valid_1']['ndcg@100']
    val_ndcg_200 = gbm.best_score['valid_1']['ndcg@200']
    
    gbm.save_model(os.path.join(save_dir, "model.txt"))
    joblib.dump(gbm, os.path.join(save_dir, "model.pkl"))
    if epoch <= 0:
        epoch = gbm.best_iteration

    pred_y_pred = pred(gbm, pred_x, pred_groups)
    train_y_pred = pred(gbm, train_x, splited_train_gps)
    splited_train_set["pred"] = train_y_pred
    splited_train_set.sort_values(by="pred", inplace=True, ascending=False)
    # if pred_mode:
    splited_train_set[["code", "code_name", "pred", label, "price"]].to_csv(os.path.join(save_dir, "train_set_EPOCH_{}.csv".format(epoch)))


    pred_dataset["pred"] = pred_y_pred
    pred_dataset.sort_values(by="pred", inplace=True, ascending=False)
    res_pred = pred_dataset[["code", "code_name", "pred", label, "price"]]
    meta = dict()
    meta["config"] = {
        "label": label,
        "train_len": train_len,
        "pred_start_day": pred_start_day,
        "pred_end_day": pred_end_day,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_data_in_leaf": min_data_in_leaf,
    }
    meta["info"] = {
        "epoch": epoch,
        "train_ndcg_3": train_ndcg_3,
        "val_ndcg_3": val_ndcg_3,
        "train_ndcg_5": train_ndcg_5,
        "val_ndcg_5": val_ndcg_5,
        "train_ndcg_10": train_ndcg_10,
        "val_ndcg_10": val_ndcg_10,
        "train_ndcg_30": train_ndcg_30,
        "val_ndcg_30": val_ndcg_30,
        "train_ndcg_50": train_ndcg_50,
        "val_ndcg_50": val_ndcg_50,
        "train_ndcg_100": train_ndcg_100,
        "val_ndcg_100": val_ndcg_100,
        "train_ndcg_200": train_ndcg_200,
        "val_ndcg_200": val_ndcg_200,
    }
    date = to_int_date(res_pred.index[0])
    
    save_file = f"{date}_T10_{val_ndcg_10}_T30_{val_ndcg_30}_T50_{val_ndcg_50}_T100_{val_ndcg_100}.csv"
    res_pred.to_csv(os.path.join(save_dir, save_file))
        
    json.dump(meta, open(os.path.join(save_dir, "meta.json"), 'w'), indent=4)
    
def join_style_info():
    generate_ltr_data()
    generate_style_learning_info()
    generate_ltr_data()

def prepare_data(update=False):
    if update:
        fetch_daily()
    remove_dir(EXP_RANK_DIR)
    remove_dir(EXP_RANK_PRED_DIR)
    inject_features()
    inject_labels()
    generate_ltr_data()
    join_style_info()
    

def get_n_val_day(label):
    if "y_rank_1d_label" in label:
        n_day = 1
    else:
        assert False
    return n_day
