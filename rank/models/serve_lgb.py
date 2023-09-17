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
    trade_days = get_trade_days()
    test_n_day = 10
    opt_points = [("y_2_d_high_rank_20%", 120, 15, 7, 5), ("y_2_d_ret_rank_20%", 60, 3, 9, 3), ("y_next_1d_close_2d_open_rate_rank_10%", 120, 15, 9, 5), ("y_next_1d_close_2d_open_rate_rank_10%", 120, 7, 3, 21)]
    
    for label, train_len, num_leaves, max_depth, min_data_in_leaf in opt_points:
        n_day = get_n_val_day(label)
        print(len(argvs))
        
        for train_val_split_day in trade_days[-test_n_day-2*n_day:-2*n_day]:
            train_start_day = to_date(get_offset_trade_day(train_val_split_day,
                                                        -train_len))
            train_end_day = to_date(get_offset_trade_day(train_val_split_day, 0))
            val_start_day = to_date(get_offset_trade_day(train_val_split_day, 1))
            val_end_day = to_date(get_offset_trade_day(train_val_split_day, n_day))
            argvs.append([
                features, label, train_start_day, train_end_day, val_start_day,
                val_end_day, n_day, train_len, num_leaves, max_depth, min_data_in_leaf, True
            ])

    np.random.shuffle(argvs)

    pool = Pool(1)
    pool.imap_unordered(train_lightgbm, argvs)
    pool.close()
    pool.join()
