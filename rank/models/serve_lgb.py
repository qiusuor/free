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
from rank.models.lgb_core import *


if __name__ == "__main__":
    
    # prepare_data()
    features = get_feature_cols()
    label = "y_2_d_high_rank_10%"
    argvs = []
    trade_days = get_trade_days(update=False)
    test_n_day = 10
    opt_points = [
        ("y_2_d_high_rank_20%_safe_1d%", 180, 31, 7, 3, 144), 
        ("y_2_d_high_rank_20%", 120, 63, 9, 3, 126), 
        ("y_2_d_high_rank_20%", 180, 63, 7, 11, 48), 
        # ("y_next_1d_close_2d_open_rate_rank_10%", 120, 15, 9, 5, 254), 
        # ("y_next_1d_close_2d_open_rate_rank_10%", 120, 3, 3, 41, 570), 
        # ("y_2_d_close_high_rank_10%", 50, 15, 9, 21, 106), 
        # ("y_2_d_close_high_rank_30%", 30, 7, 3, 21, 99),
    ]
    
    for label, train_len, num_leaves, max_depth, min_data_in_leaf, epoch in opt_points:
        n_day = get_n_val_day(label)
        # print(len(argvs))
        train_val_split_day = trade_days[-n_day-1]
        
        train_start_day = to_date(get_offset_trade_day(train_val_split_day,
                                                    -train_len))
        train_end_day = to_date(get_offset_trade_day(train_val_split_day, 0))
        val_start_day = to_date(get_offset_trade_day(train_val_split_day, 1))
        val_end_day = to_date(get_offset_trade_day(train_val_split_day, n_day))
        argvs.append([
            features, label, train_start_day, train_end_day, val_start_day,
            val_end_day, n_day, train_len, num_leaves, max_depth, min_data_in_leaf, epoch
        ])
    #     print(train_start_day, train_end_day, val_start_day, val_end_day)
    # exit(0)
    np.random.shuffle(argvs)
    # print(argvs[0])
    # train_lightgbm(argvs[0])
    pool = Pool(1)
    pool.imap_unordered(train_lightgbm, argvs)
    pool.close()
    pool.join()
