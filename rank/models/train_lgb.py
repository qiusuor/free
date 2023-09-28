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
    
    search_labels = [
        # "y_next_1d_close_2d_open_rate_rank_10%",
        # "y_next_1d_close_2d_open_rate_rank_20%",
        # "y_next_1d_close_2d_open_rate_rank_30%",
        # "y_next_1d_close_2d_open_rate_rank_50%",
        
        # "y_2_d_close_high_rank_10%",
        # "y_2_d_close_high_rank_20%",
        # "y_2_d_close_high_rank_30%",
        # "y_2_d_close_high_rank_50%",
        
        "y_2_d_high_rank_10%_safe_1d",
        "y_2_d_high_rank_20%_safe_1d",
        "y_2_d_high_rank_30%_safe_1d",
        # "y_2_d_high_rank_50%_safe_1d",
        
        "y_2_d_high_rank_10%",
        "y_2_d_high_rank_20%",
        "y_2_d_high_rank_30%",
        # "y_2_d_high_rank_50%",
        
        # "y_2_d_ret_rank_10%",
        # "y_2_d_ret_rank_20%",
        # "y_2_d_ret_rank_30%",
        # "y_2_d_ret_rank_50%",
    
    ]
    
    features = get_feature_cols()
    trade_days = get_trade_days(update=False)
    test_n_day = TEST_N_LAST_DAY
    argvs = []
    
    for label in search_labels:
        for num_leaves in [3, 7, 15, 31, 63]:
            for min_data_in_leaf in [3, 5, 11, 21, 41]:
                for max_depth in [3, 7, 9, 12]:
                    if 2**max_depth <= num_leaves: continue
                    for train_len in [30, 50, 120, 180]:
                        n_day = get_n_val_day(label)
                        
                        print(len(argvs))
                        # print(trade_days[-test_n_day-2*n_day:-2*n_day])
                        
                        for train_val_split_day in trade_days[-test_n_day-2*n_day:-2*n_day]:
                            train_start_day = to_date(get_offset_trade_day(train_val_split_day,
                                                                        -train_len))
                            train_end_day = to_date(get_offset_trade_day(train_val_split_day, 0))
                            val_start_day = to_date(get_offset_trade_day(train_val_split_day, 1))
                            # print(val_start_day)
                            val_end_day = to_date(get_offset_trade_day(train_val_split_day, n_day))
                            argvs.append([
                                features, label, train_start_day, train_end_day, val_start_day,
                                val_end_day, n_day, train_len, num_leaves, max_depth, min_data_in_leaf, -1
                            ])
                        # exit(0)
                        

    # exit(0)
    np.random.shuffle(argvs)
    # print(argvs[0])
    # train_lightgbm(argvs[0])
    pool = Pool(16)
    pool.imap_unordered(train_lightgbm, argvs)
    pool.close()
    pool.join()
