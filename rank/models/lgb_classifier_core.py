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
from matplotlib import pyplot as plt
import shutil
import json
import gc
from data.generate_ltr_data import generate_ltr_data
from data.generate_style_leaning_feature import generate_style_learning_info
from data.inject_labels import inject_labels
import bisect

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

def style_filter(train_set, val_set):
    filed = "style_feat_y_open_close_mean_limit_up_high"
    thresh = [-1e6, -0.036716, 0.000000, 0.022565, 1e6]
    mean_val = val_set[filed].mean()
    cut_index = bisect.bisect_left(thresh, mean_val)
    left_bound, right_bound = thresh[cut_index], thresh[cut_index+1]
    train_set = train_set[(train_set[filed] >= left_bound) & (train_set[filed] <= right_bound)]
    return train_set
    
def train_lightgbm(argv):
    features, label, train_start_day, train_end_day, val_start_day, val_end_day, n_day, train_len, num_leaves, max_depth, min_data_in_leaf, cache_data, epoch = argv
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {"average_precision"},
        'num_leaves': num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        'learning_rate': 0.05,
        'feature_fraction': 0.99,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'verbose': 1,
        "train_metric": True,
        "max_depth": max_depth,
        "num_iterations": 5000,
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
    root_dir = EXP_CLS_PRED_DIR if pred_mode else EXP_CLS_DIR
    save_dir = "{}/{}/{}/{}".format(root_dir, label, param_des, to_int_date(val_start_day))
    if os.path.exists(os.path.join(save_dir, "meta.json")):
        with open(os.path.join(save_dir, "meta.json")) as f:
            meta = json.load(f)
            if not np.isnan(meta["last_val"]["auc"]):
                return
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    make_dir(save_dir)
    make_dir(cache_data)
    if os.path.exists(cache_data):
        with open(cache_data, 'rb') as f:
            dataset = cPickle.load(f)
    else:
        dataset = []
        for path in main_board_stocks():
            df = joblib.load(path)
            df["date"] = df.index
            df = df[is_limit_up(df)]
            dataset.append(df)
        dataset = pd.concat(dataset, axis=0)
        with open(cache_data, 'wb') as f:
            cPickle.dump(dataset, f)
    train_dataset = dataset[(dataset.date >= train_start_day) & (dataset.date <= train_end_day)]
    # train_dataset = train_dataset[train_dataset.style_feat_open_close_mean_limit_up_high_pre_no_limit_up > -0.045655402518364696]
    val_dataset = dataset[(dataset.date >= val_start_day) & (dataset.date <= val_end_day)]
    del dataset
    # train_dataset = style_filter(train_dataset, val_dataset)
    gc.collect()
    train_x, train_y = train_dataset[features], train_dataset[label]
    val_x, val_y = val_dataset[features], val_dataset[label]
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(val_x, val_y, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=(lgb_train, lgb_eval),
                    # categorical_feature=["industry"]
                    )

    gbm.save_model(os.path.join(save_dir, "model.txt"))
    joblib.dump(gbm, os.path.join(save_dir, "model.pkl"))
    if epoch <= 0:
        epoch = gbm.best_iteration

    val_y_pred = gbm.predict(val_x, num_iteration=epoch)
    train_y_pred = gbm.predict(train_x, num_iteration=epoch)
    train_dataset["pred"] = train_y_pred
    train_dataset.sort_values(by="pred", inplace=True, ascending=False)
    train_ap = round(average_precision_score(train_dataset[label], train_dataset.pred), 2)
    train_auc = round(roc_auc_score(train_dataset[label], train_dataset.pred), 2)
    if pred_mode:
        train_dataset[["code", "code_name", "pred", label, "y_next_1d_close_rate", f"y_next_{2}_d_high_ratio", f"y_next_{2}_d_low_ratio", "y_next_{}_d_ret".format(2), "y_next_1d_close_2d_open_rate", "price"]].to_csv(os.path.join(save_dir, "train_set_EPOCH_{}_AP_{}_AUC_{}.csv".format(epoch, train_ap, train_auc)))
    fpr, tpr, thresh = roc_curve(val_y, val_y_pred)
    val_auc = auc(fpr, tpr)
    val_ap = average_precision_score(val_y, val_y_pred)
    plt.clf()
    plt.plot(fpr,
             tpr,
             'k--',
             label='ROC (area = {0:.2f})'.format(val_auc),
             lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, "roc_curve.png"))
    val_dataset["pred"] = val_y_pred
    res_val = val_dataset[["code", "code_name", "pred", label, "y_next_1d_close_rate", f"y_next_{2}_d_high_ratio", f"y_next_{2}_d_low_ratio", "y_next_{}_d_ret".format(2), "y_next_1d_close_2d_open_rate", "price"]]
    meta = dict()
    meta["config"] = {
        "label": label,
        "train_len": train_len,
        "val_start_day": to_int_date(val_start_day),
        "val_end_day": to_int_date(val_end_day),
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_data_in_leaf": min_data_in_leaf,
    }
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
    for i, res_i in res_val.groupby("date"):
        res_i.sort_values(by="pred", inplace=True, ascending=False)
        top3_miss, top3_shot, top3_watch = topk_shot(res_i, label, k=3, watch_list=watch_list)
        top5_miss, top5_shot, top5_watch = topk_shot(res_i, label, k=5, watch_list=watch_list)
        top10_miss, top10_shot, top10_watch = topk_shot(res_i, label, k=10, watch_list=watch_list)
        fpr, tpr, thresh = roc_curve(res_i[label], res_i.pred)
        auc_score = auc(fpr, tpr)
        ap = average_precision_score(res_i[label], res_i.pred)
        meta["daily"][to_int_date(i)] = {
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
        meta["last_val"] = meta["daily"][to_int_date(i)]
        save_file = f"{to_int_date(i)}_T3_{top3_miss}_T5_{top5_miss}_T10_{top10_miss}_AP_{ap}_AUC_{auc_score}.csv"
        res_i.to_csv(os.path.join(save_dir, save_file))
    
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
    
def join_style_info():
    generate_ltr_data()
    generate_style_learning_info()
    
    
def prepare_data(update=False):
    if update:
        fetch_daily()
    remove_dir(EXP_CLS_DIR)
    remove_dir(EXP_CLS_PRED_DIR)
    inject_features()
    inject_labels()
    # inject_embedding()
    join_style_info()
    # inject_minute_feature()

def get_n_val_day(label):
    if "y_next_1d_up_to_limit" in label or "y_next_1d_close_rate" in label:
        n_day = 1
    elif "y_2_d" in label or "1d_close_2d_open" in label or "y_02" in label or "y_next_2_d" in label:
        n_day = 2
    elif "y_5_d" in label:
        n_day = 5
    else:
        assert False
    return n_day
