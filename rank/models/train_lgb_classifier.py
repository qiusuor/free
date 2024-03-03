import joblib
import numpy as np
from multiprocessing import Pool
from utils import *
from rank.models.lgb_classifier_core import *
from rank.models.agg_prediction_info import agg_prediction_info
import bisect
import platform


if __name__ == "__main__":
    
    df = joblib.load(os.path.join(DAILY_DIR, "sh.600000_d_2.pkl"))
    if "adx_2" not in df.columns:
        prepare_data(update=False)
    
    search_labels = [
        # "y_next_1d_close_2d_open_rate_rank_10%",
        # "y_next_1d_close_2d_open_rate_rank_20%",
        # "y_next_1d_close_2d_open_rate_rank_30%",
        # "y_next_1d_close_2d_open_rate_rank_50%",
        
        # "y_2_d_close_high_rank_10%",
        # "y_2_d_close_high_rank_20%",
        # "y_2_d_close_high_rank_30%",
        # "y_2_d_close_high_rank_50%",
        
        # "y_02_103",
        # "y_02_105",
        # "y_02_107",
        # "y_02_109",
        # "y_next_1d_up_to_limit",
        
        "y_next_2_d_ret_00",
        "y_next_2_d_ret_01",
        "y_next_2_d_ret_02",
        "y_next_2_d_ret_03",
        "y_next_2_d_ret_04",
        "y_next_2_d_ret_05",
        "y_next_2_d_ret_07",
        "y_next_2_d_ret_095",
        "y_next_2_d_ret_12",
        "y_next_2_d_ret_15",
        "y_next_2_d_ret_17",
        
        # "y_2_d_high_rank_10%_safe_1d",
        # "y_2_d_high_rank_20%_safe_1d",
        # "y_2_d_high_rank_30%_safe_1d",
        # "y_2_d_high_rank_50%_safe_1d",
        
        # "y_2_d_high_rank_10%",
        # "y_2_d_high_rank_20%",
        # "y_2_d_high_rank_30%",
        # "y_2_d_high_rank_50%",
        
        # "y_2_d_ret_rank_10%",
        # "y_2_d_ret_rank_20%",
        # "y_2_d_ret_rank_30%",
        # "y_2_d_ret_rank_50%",
        
        # "y_2_d_ret_rank_10%_safe_1d",
        # "y_2_d_ret_rank_20%_safe_1d",
        # "y_2_d_ret_rank_30%_safe_1d",
        # "y_2_d_ret_rank_50%_safe_1d",
    
    ]
    
    features = get_feature_cols()
    trade_days = get_trade_days(update=False)
    trunc_index = bisect.bisect_right(trade_days, SEARCH_END_DAY)
    trade_days = trade_days[:trunc_index]
    cache_data = EXP_CLS_DATA_CACHE.format(trade_days[-1])
    val_n_day = VAL_N_LAST_DAY
    argvs = []
    
    
    for label in search_labels:
        # for train_len, num_leaves, max_depth, min_data_in_leaf in [[250, 7, 3, 21]]:
        
        for num_leaves in [3, 7, 15, 31, 63]:
            for min_data_in_leaf in [5, 11, 21, 41, 81]:
                for max_depth in [3, 7, 9, 12]:
                    if 2**max_depth <= num_leaves: continue
                    # for train_len in [250]:
                    for train_len in [3, 5, 10, 20, 30, 50, 120, 240]:
                        n_day = get_n_val_day(label)
                        
                        print(len(argvs))
                        # print(trade_days[-val_n_day-2*n_day:-2*n_day])
                        
                        for train_val_split_day in trade_days[-val_n_day-2*n_day-val_delay_day:-n_day-val_delay_day]:
                            # print(train_val_split_day)
                            train_start_day = to_date(get_offset_trade_day(train_val_split_day,
                                                                        -train_len+1))
                            train_end_day = to_date(get_offset_trade_day(train_val_split_day, 0))
                            
                            val_start_day = to_date(get_offset_trade_day(train_val_split_day, 1 + val_delay_day))
                            val_end_day = to_date(get_offset_trade_day(to_int_date(val_start_day), n_day-1))
                            argv = [
                                features, label, train_start_day, train_end_day, val_start_day,
                                val_end_day, n_day, train_len, num_leaves, max_depth, min_data_in_leaf, cache_data, -1
                            ]
                            print(argv)
                            train_lightgbm(argv)
                            print("Generate cache file this time, try again.")
                            exit(0)
                            if not os.path.exists(cache_data):
                                print(argv)
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
    agg_prediction_info(last_n_day=val_n_day)
    
