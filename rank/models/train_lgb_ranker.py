import joblib
import numpy as np
from multiprocessing import Pool
from utils import *
from rank.models.lgb_ranker_core import *
from rank.models.agg_prediction_info import agg_prediction_info
import bisect
import platform


if __name__ == "__main__":
    
    df = joblib.load(os.path.join(DAILY_DIR, "sh.600000_d_2.pkl"))
    if "adx_2" not in df.columns:
        prepare_data(update=False)
    
    search_labels = [
        "y_rank_1d_label",
    
    ]
    
    features = get_feature_cols()
    trade_days = get_trade_days(update=False)
    trunc_index = bisect.bisect_right(trade_days, SEARCH_END_DAY)
    trade_days = trade_days[:trunc_index]
    cache_data = EXP_RANK_DATA_CACHE.format(trade_days[-1])
    val_n_day = VAL_N_LAST_DAY
    argvs = []
    
    
    for label in search_labels:
        for num_leaves in [5, 7, 15, 31, 63]:
            for min_data_in_leaf in [3, 5, 11, 21, 41, 81]:
                for max_depth in [3, 5, 7, 9]:
                    if 2**max_depth <= num_leaves: continue
                    for train_len in [250]:
                    # for train_len in [30, 50, 120, 180]:
                        n_day = get_n_val_day(label)
                        
                        print(len(argvs))
                        # print(trade_days[-val_n_day-2*n_day:-2*n_day])
                        
                        for train_val_split_day in trade_days[-val_n_day-n_day-val_delay_day:-n_day-val_delay_day]:
                            train_start_day = to_date(get_offset_trade_day(train_val_split_day,
                                                                        -train_len))
                            train_end_day = to_date(get_offset_trade_day(train_val_split_day, 0))
                            
                            val_start_day = to_date(get_offset_trade_day(train_val_split_day, 1 + val_delay_day))
                            val_end_day = to_date(get_offset_trade_day(to_int_date(val_start_day), n_day-1))
                            argv = [
                                features, label, train_start_day, train_end_day, val_start_day,
                                val_end_day, n_day, train_len, num_leaves, max_depth, min_data_in_leaf, cache_data, -1
                            ]
                            # print(argv)
                            # exit(0)
                            if not os.path.exists(cache_data):
                                # print(argv)
                                train_lightgbm(argv)
                                print("Generate cache file this time, try again.")
                                exit(0)
                            argvs.append(argv)
                        # exit(0)
                        

    np.random.shuffle(argvs)
    pool = Pool(16 if "Linux" in platform.platform() else 2)
    pool.imap_unordered(train_lightgbm, argvs)
    pool.close()
    pool.join()
    # agg_prediction_info(ana_dir=EXP_RANK_DIR, last_n_day=val_n_day)
    
