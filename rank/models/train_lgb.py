import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_curve, auc
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


def train_lightgbm(features, label, train_start_day, train_val_split_day, val_end_day):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {"auc", "average_precision"},
        'num_leaves': 1023,
        "min_data_in_leaf": 7,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 2,
        'verbose': 1,
        "train_metric": True,
        "max_depth": 21,
        "num_iterations": 5000,
        "early_stopping_rounds": 100,
        "device": 'gpu',
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "min_gain_to_split": 0,
        "num_threads": 256,
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

        train_dataset.append(df[train_start_day:train_val_split_day])
        val_dataset.append(df[train_val_split_day:val_end_day])
        
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
                    )
    gbm.save_model("{}_{}_{}.txt".format(label, train_val_split_day, val_end_day))
    # joblib.dump(gbm, "{}_{}_{}.pkl".format(label, train_val_split_day, val_end_day))
    
    
if __name__ == "__main__":
    # fetch_daily()
    # inject_features()
    # inject_labels()
    # injecto_joint_label()
    
    features = get_feature_cols()
    label = "y_2_d_high_rank_30%"
    anchor = 20230801
    
    train_val_split_day = to_date(anchor)
    train_start_day = to_date(get_offset_trade_day(anchor, -5))
    val_end_day = to_date(get_offset_trade_day(anchor, 5))
    
    train_lightgbm(features, label, train_start_day, train_val_split_day, val_end_day)