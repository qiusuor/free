import numpy as np
from multiprocessing import Pool
from utils import *
import json
from rank.models.lgb_classifier_core import *
from rank.models.agg_prediction_info import agg_prediction_info
import platform
import bisect
import math

def parse_best_opts():
    opts = set()
    fields = ["sharp_exp"]
    topk = 20
    # fields = ["topk_miss_exp", "topk_ret_exp", "sharp_exp", "high_exp"]
    
    with open(os.path.join(EXP_CLS_DIR, "agg_info.json")) as f:
        train_agg_info = json.load(f)
        
        for field in fields:
            exps = train_agg_info["Top-3"][field]
            for k in range(min(topk, len(exps))):
                best_exp = exps[list(exps)[k]]
                label = best_exp["label"]
                config = list(map(int, best_exp["exp_config"].split("_")))
                epoch = math.ceil(best_exp["avg_epoch"])
                opt = (label, *config, epoch)
                opts.add(opt)
        
    return opts
    
if __name__ == "__main__":
    
    # prepare_data(update=False)
    features = get_feature_cols()
    argvs = []
    trade_days = get_trade_days(update=False)
    trunc_index = bisect.bisect_right(trade_days, SEARCH_END_DAY)
    trade_days = trade_days[:trunc_index]

    cache_data = EXP_CLS_DATA_CACHE.format(trade_days[-1])
    opt_points = parse_best_opts()
    
    for label, train_len, num_leaves, max_depth, min_data_in_leaf, epoch in opt_points:
        # eval on multi run
        n_day = get_n_val_day(label)
        for k in range(TEST_N_LAST_DAY + n_day):
            # print(len(argvs))
            train_val_split_day = trade_days[-n_day-1-k]
            
            train_start_day = to_date(get_offset_trade_day(train_val_split_day,
                                                        -train_len-val_delay_day))
            train_end_day = to_date(get_offset_trade_day(train_val_split_day, 0-val_delay_day))
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
    pool = Pool(8 if "Linux" in platform.platform() else 2)
    pool.imap_unordered(train_lightgbm, argvs)
    pool.close()
    pool.join()
    agg_prediction_info(EXP_CLS_PRED_DIR, last_n_day=TEST_N_LAST_DAY)
    
