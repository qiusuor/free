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
from data.fetch import fetch_daily
from data.inject_features import inject_features
from data.inject_labels import inject_labels
from matplotlib import pyplot as plt
import shutil
import json


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
    return miss_cnt, shot_cnt, watches


def train_lightgbm(argv):
    features, label, train_start_day, train_end_day, val_start_day, val_end_day, n_day, train_len, num_leaves, max_depth, min_data_in_leaf = argv
    param_des = "_".join([str(train_len), str(num_leaves), str(max_depth), str(min_data_in_leaf)])
    save_dir = "{}/{}/{}/{}".format(EXP_DIR, label, param_des, to_int_date(val_start_day))
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    make_dir(save_dir)
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
        "early_stopping_rounds": 100,
        # "device": 'gpu',
        # "gpu_platform_id": 0,
        # "gpu_device_id": 0,
        "min_gain_to_split": 0,
        "num_threads": 16,
    }
    # print(params)

    train_dataset = []
    val_dataset = []
    for file in os.listdir(DAILY_DIR):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        df = joblib.load(path)
        if df.isST[-1]:
            continue
        if "code_name" not in df.columns or not isinstance(df.code_name[-1], str) or "ST" in df.code_name[-1] or "st" in df.code_name[-1] or "sT" in df.code_name[-1]:
            continue

        train_dataset.append(df[train_start_day:train_end_day])
        val_dataset.append(df[val_start_day:val_end_day])

    train_dataset = pd.concat(train_dataset, axis=0)
    val_dataset = pd.concat(val_dataset, axis=0)

    train_x, train_y = train_dataset[features], train_dataset[label]
    val_x, val_y = val_dataset[features], val_dataset[label]
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(val_x, val_y, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=5000,
                    valid_sets=(lgb_train, lgb_eval),
                    # categorical_feature=["industry"]
                    )

    gbm.save_model(os.path.join(save_dir, "model.txt"))
    joblib.dump(gbm, os.path.join(save_dir, "model.pkl"))
    epoch = gbm.best_iteration

    val_y_pred = gbm.predict(val_x, num_iteration=epoch)
    train_y_pred = gbm.predict(train_x, num_iteration=epoch)
    train_dataset["pred"] = train_y_pred
    train_dataset.sort_values(by="pred", inplace=True, ascending=False)
    train_ap = round(average_precision_score(train_dataset[label], train_dataset.pred), 2)
    train_auc = round(roc_auc_score(train_dataset[label], train_dataset.pred), 2)
    # train_dataset[["code", "code_name", "pred", label, "y_next_1d_close_2d_open_rate", f"y_next_{n_day}_d_high_ratio", f"y_next_{n_day}_d_low_ratio", "price"]].to_csv(os.path.join(save_dir, "train_set_EPOCH_{}_AP_{}_AUC_{}.csv".format(epoch, train_ap, train_auc)))
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
    res_val = val_dataset[["code", "code_name", "pred", label, "y_next_1d_close_2d_open_rate", f"y_next_{n_day}_d_high_ratio", f"y_next_{n_day}_d_low_ratio", "price"]]
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
    watch_list = [f"y_next_{n_day}_d_high_ratio", f"y_next_{n_day}_d_low_ratio", "y_next_1d_close_2d_open_rate"]
    
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
            "top10_watch": top10_watch
        }
        if not np.isnan(ap) and ap > 0:
            meta["last_val"] = {
                "auc": auc_score,
                "ap": ap,
                "top3_watch": top3_watch,
                "top5_watch": top5_watch,
                "top10_watch": top10_watch
            }
        save_file = f"{to_int_date(i)}_T3_{top3_miss}_T5_{top5_miss}_T10_{top10_miss}_AP_{ap}_AUC_{auc_score}.csv"
        res_i.to_csv(os.path.join(save_dir, save_file))
    json.dump(meta, open(os.path.join(save_dir, "meta.json"), 'w'), indent=4)

def prepare_data(update=True):
    if update:
        fetch_daily()
    inject_features()
    inject_labels()

def get_n_val_day(label):
    if "y_5_d" in label:
        n_day = 5
    elif "y_10_d" in label:
        n_day = 10
    elif "y_2_d" in label or "1d_close_2d_open" in label:
        n_day = 2
    else:
        assert False
    return n_day