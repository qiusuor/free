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


if __name__ == "__main__":
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {"auc", "average_precision"},
        'num_leaves': 15,
        "min_data_in_leaf": 3,
        'learning_rate': 0.05,
        'feature_fraction': 0.999,
        'bagging_fraction': 0.9,
        'bagging_freq': 2,
        'verbose': 1,
        "train_metric": True,
        "max_depth": 7,
        "num_iterations": 5000,
        "early_stopping_rounds": 100,
        # "device": 'gpu',
        # "gpu_platform_id": 0,
        # "gpu_device_id": 0,
        "min_gain_to_split": 0,
        "num_threads": 128,
    }
    
    train_dataset = np.load("three_days_increase_train_lgb.npz") 
    val_dataset = np.load("three_days_increase_val_lgb.npz")
    
    train_x, train_y = train_dataset["x"], train_dataset["y"]
    val_x, val_y = val_dataset["x"], val_dataset["y"]
    
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(val_x, val_y, reference=lgb_train)
    
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=5000,
                    valid_sets=(lgb_train, lgb_eval),
                    )