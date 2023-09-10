import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_curve, auc, average_precision_score
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
from joblib import load, dump
import _pickle as cPickle
from multiprocessing import Process
import hashlib
import copy
import bisect
import random
from utils import *
from data.fetch import fetch_daily
from data.inject_features import inject_features
from data.inject_labels import inject_labels
from data.inject_joint_label import injecto_joint_label
from matplotlib import pyplot as plt
import shutil

def topk_shot(label, k=10):
    gt_labels = label.values[:k]
    shot_cnt = 0
    miss_cnt = 0
    for label in gt_labels:
        if label:
            shot_cnt += 1
        else:
            miss_cnt += 1
    return miss_cnt, shot_cnt


def train_lightgbm(features, label, train_start_day, train_end_day,
                   val_start_day, val_end_day):
    save_dir = "{}/{}/{}".format(EXP_DIR, label, to_int_date(val_start_day))
    if os.path.exists(save_dir):
        shutil.rmtree(os.path.dirname(save_dir))
    make_dir(save_dir)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {"auc", "average_precision"},
        'num_leaves': 15,
        "min_data_in_leaf": 7,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 2,
        'verbose': 1,
        "train_metric": True,
        "max_depth": 5,
        "num_iterations": 5000,
        "early_stopping_rounds": 100,
        "device": 'gpu',
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "min_gain_to_split": 0,
        "num_threads": 16,
    }

    train_dataset = []
    val_dataset = []
    for file in tqdm(os.listdir(DAILY_DIR)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        df = joblib.load(path)

        train_dataset.append(df[train_start_day:train_end_day])
        val_dataset.append(df[val_start_day:val_end_day])

    train_dataset = pd.concat(train_dataset, axis=0)
    val_dataset = pd.concat(val_dataset, axis=0)

    train_x, train_y = train_dataset[features], train_dataset[label]
    val_x, val_y = val_dataset[features], val_dataset[label]
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(val_x, val_y, reference=lgb_train)

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=(lgb_train, lgb_eval),
        categorical_feature=["industry"]
    )


    gbm.save_model(os.path.join(save_dir, "model.txt"))
    joblib.dump(gbm, os.path.join(save_dir, "model.pkl"))

    val_y_pred = gbm.predict(val_x, num_iteration=gbm.best_iteration)
    train_y_pred = gbm.predict(train_x, num_iteration=gbm.best_iteration)
    train_dataset["pred"] = train_y_pred
    train_dataset[["pred", "code", "price",
                   label]].to_csv(os.path.join(save_dir, "train_set.cs v"))
    fpr, tpr, thresh = roc_curve(val_y, val_y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,
             tpr,
             'k--',
             label='ROC (area = {0:.2f})'.format(roc_auc),
             lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, "roc_curve.png"))

    val_dataset["pred"] = val_y_pred
    res = val_dataset[["pred", "code", "price", label]]
    for i, res_i in res.groupby("date"):
        res_i.sort_values(by="pred", inplace=True, ascending=False)
        top3_miss, top3_shot = topk_shot(res_i[label], k=3)
        top5_miss, top5_shot = topk_shot(res_i[label], k=5)
        top10_miss, top10_shot = topk_shot(res_i[label], k=10)
        fpr, tpr, thresh = roc_curve(res_i[label], res_i.pred)
        auc_score = auc(fpr, tpr)
        ap = average_precision_score(res_i[label], res_i.pred)
        save_file = f"{to_int_date(i)}_T3_{top3_miss}_T5_{top5_miss}_T10_{top10_miss}_AP_{ap}_AUC_{auc_score}.csv"
        res_i.to_csv(os.path.join(save_dir, save_file))

def prepare_data():
    fetch_daily()
    inject_labels()
    injecto_joint_label()
    inject_features()

if __name__ == "__main__":
    # prepare_data()

    features = get_feature_cols()
    label = "y_2_d_high_rank_30%"

    train_val_split_day = 20230821
    train_start_day = to_date(get_offset_trade_day(train_val_split_day, -5))
    train_end_day = to_date(get_offset_trade_day(train_val_split_day, -1))
    val_start_day = to_date(train_val_split_day)
    val_end_day = to_date(get_offset_trade_day(train_val_split_day, 0))

    train_lightgbm(features, label, train_start_day, train_end_day,
                   val_start_day, val_end_day)

