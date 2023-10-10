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
from matplotlib import pyplot as plt
import shutil
import json
from rank.models.lgb_core import *
from rank.models.agg_prediction_info import agg_prediction_info
import platform
import bisect
import math

def parse_best_opts():
    opts = []
    
    with open(os.path.join(EXP_DIR, "agg_info.json")) as f:
        train_agg_info = json.load(f)
        sharp_exp = train_agg_info["Top-3"]["sharp_exp"]
        best_sharp_exp = sharp_exp[list(sharp_exp)[0]]
        label = best_sharp_exp["label"]
        config = list(map(int, best_sharp_exp["exp_config"].split("_")))
        epoch = math.ceil(best_sharp_exp["avg_epoch"])
        opt = [label, *config, epoch]
        opts.append(opt)
    
    return opts
    
if __name__ == "__main__":
    
    # prepare_data(update=False)
    features = get_feature_cols()
    label = "y_2_d_high_rank_10%"
    argvs = []
    trade_days = get_trade_days(update=False)
    trunc_index = bisect.bisect_right(trade_days, SEARCH_END_DAY)
    trade_days = trade_days[:trunc_index]

    cache_data = EXP_DATA_CACHE.format(trade_days[-1])
    opt_points = parse_best_opts()
    
    for label, train_len, num_leaves, max_depth, min_data_in_leaf, epoch in opt_points:
        # eval on multi run
        for k in range(TEST_N_LAST_DAY + 2):
            n_day = get_n_val_day(label)
            # print(len(argvs))
            train_val_split_day = trade_days[-n_day-1-k]
            
            train_start_day = to_date(get_offset_trade_day(train_val_split_day,
                                                        -train_len))
            train_end_day = to_date(get_offset_trade_day(train_val_split_day, 0))
            val_start_day = to_date(get_offset_trade_day(train_val_split_day, 1))
            val_end_day = to_date(get_offset_trade_day(train_val_split_day, n_day))
            argv = [
                features, label, train_start_day, train_end_day, val_start_day,
                val_end_day, n_day, train_len, num_leaves, max_depth, min_data_in_leaf, cache_data, epoch
            ]
            if not os.path.exists(cache_data):
                # print(argv)
                train_lightgbm(argv)
                print("Generate cache file this time, try again.")
                exit(0)
            argvs.append(argv)
    #     print(train_start_day, train_end_day, val_start_day, val_end_day)
    # print(argvs[0])
    # train_lightgbm(argvs[0])
    # exit(0)

    np.random.shuffle(argvs)
    pool = Pool(32 if "Linux" in platform.platform() else 1)
    pool.imap_unordered(train_lightgbm, argvs)
    pool.close()
    pool.join()
    agg_prediction_info(EXP_PRED_DIR)
    
