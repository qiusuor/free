from config import *
from utils import *
import os
import json
import numpy as np

def to_dict(sorted_items):
    keys, vals = [], []
    for key, val in sorted_items:
        keys.append(key)
        vals.append(val)
    return dict(zip(keys, vals))
        
def agg_prediction_info(ana_dir=EXP_DIR, last_n_day=TEST_N_LAST_DAY):
    all_agg_result = dict()
    for topk in [3, 5, 10]:
        agg_result = all_agg_result["Top-{}".format(topk)] = dict()
        agg_result["sharp_exp"] = dict()
        agg_result["close_open_strategy"] = dict()
        agg_result["close_high_strategy"] = dict()
        max_high_rate = -np.inf
        max_high_rate_config = None
        max_sharp_rate = -np.inf
        max_sharp_rate_config = None
        max_avg_ap = -np.inf
        max_avg_ap_config = None
        max_next_1d_close_2d_open_gain = -np.inf
        max_next_1d_close_2d_open_gain_config = None
        max_avg_close_high = -np.inf
        max_avg_close_high_config = None
        max_avg_close_sharp_rate = -np.inf
        max_avg_close_sharp_rate_config = None
        for label in tqdm(os.listdir(ana_dir)):
            if "5_d" in label: continue
            close_open_strategy = "y_next_1d_close_2d_open_rate_rank" in label
            close_high_strategy = "y_2_d_close_high" in label
            if not label.startswith("y") and not label.startswith("dy"): continue
            label_dir = os.path.join(ana_dir, label)
            for exp_config in os.listdir(label_dir):
                configured_exp_dir = os.path.join(label_dir, exp_config)
                if not os.path.isdir(configured_exp_dir): continue
                epochs, last_aps, last_aucs, topk_high_means, topk_low_means, topk_sharp_means, topk_gain_means, topk_close_high_means, topk_close_low_means, topk_close_sharps, topk_1d_close_means = [], [], [], [], [], [], [], [], [], [], []
                topk_misses, topk_ret_means = [], []
                if len(os.listdir(configured_exp_dir)) < test_last_n_day: continue
                finished = True
                val_start_days = list(filter(lambda x: os.path.isdir(os.path.join(configured_exp_dir, x)), os.listdir(configured_exp_dir)))
                if len(val_start_days) > test_last_n_day:
                    val_start_days = list(sorted(val_start_days, key=lambda x:int(x)))[-test_last_n_day-2:-2]
                # print(val_start_days)
                feature_importance = 0
                
                for val_start_day in val_start_days:
                    val_start_day_dir = os.path.join(configured_exp_dir, val_start_day)
                    # print(val_start_day_dir)
                    
                    if not os.path.isdir(val_start_day_dir): continue
                    if not os.path.exists(os.path.join(val_start_day_dir, "meta.json")):
                        finished = False
                        break
                    model = joblib.load(os.path.join(val_start_day_dir, "model.pkl"))
                    feature_importance += model.feature_importance()
                    feature_name = model.feature_name()
                    meta = json.load(open(os.path.join(val_start_day_dir, "meta.json")))
                    epoch = meta["info"]["epoch"]
                    last_ap = meta["mean_val"]["ap"]
                    last_auc = meta["mean_val"]["auc"]
                    topk_high_mean = meta["mean_val"]["top{}_watch".format(topk)]["y_next_2_d_high_ratio_topk_{}_mean".format(topk)]
                    topk_low_mean = meta["mean_val"]["top{}_watch".format(topk)]["y_next_2_d_low_ratio_topk_{}_mean".format(topk)]
                    topk_gain_mean = meta["mean_val"]["top{}_watch".format(topk)]["y_next_1d_close_2d_open_rate_topk_{}_mean".format(topk)] if "y_next_1d_close_2d_open_rate_topk_{}_mean".format(topk) in meta["mean_val"]["top{}_watch".format(topk)] else 0
                    topk_close_high_mean = meta["mean_val"]["top{}_watch".format(topk)]["y_next_2_d_close_high_ratio_topk_{}_mean".format(topk)] if "y_next_2_d_close_high_ratio_topk_{}_mean".format(topk) in meta["mean_val"]["top{}_watch".format(topk)] else 0
                    topk_close_low_mean = meta["mean_val"]["top{}_watch".format(topk)]["y_next_2_d_close_low_ratio_topk_{}_mean".format(topk)] if "y_next_2_d_close_low_ratio_topk_{}_mean".format(topk) in meta["mean_val"]["top{}_watch".format(topk)] else 0
                    topk_1d_close_mean = meta["mean_val"]["top{}_watch".format(topk)]["y_next_1d_close_rate_topk_{}_mean".format(topk)] if "y_next_1d_close_rate_topk_{}_mean".format(topk) in meta["mean_val"]["top{}_watch".format(topk)] else 0
                    
                    topk_miss = meta["mean_val"]["top{}_miss".format(topk)]
                    topk_ret_mean = meta["mean_val"]["top{}_watch".format(topk)]["y_next_2_d_ret_topk_{}_mean".format(topk)]
                    
                    topk_sharp_mean = topk_high_mean * topk_low_mean
                    topk_close_sharp = topk_close_high_mean * topk_close_low_mean
                    
                    epochs.append(epoch)
                    last_aps.append(last_ap)
                    last_aucs.append(last_auc)
                    topk_high_means.append(topk_high_mean)
                    topk_low_means.append(topk_low_mean)
                    topk_sharp_means.append(topk_sharp_mean)
                    topk_gain_means.append(topk_gain_mean)
                    topk_close_high_means.append(topk_close_high_mean)
                    topk_close_low_means.append(topk_close_low_mean)
                    topk_close_sharps.append(topk_close_sharp)
                    topk_1d_close_means.append(topk_1d_close_mean)
                    topk_misses.append(topk_miss)
                    topk_ret_means.append(topk_ret_mean)
                    
                if not finished: continue
                avg_epoch = np.mean(epochs)
                avg_auc = np.mean(last_aucs)
                avg_ap = np.mean(last_aps)
                agv_high = np.mean(topk_high_means)
                avg_low = np.mean(topk_low_means)
                avg_sharp = np.mean(topk_sharp_means)
                avg_close_open = np.mean(topk_gain_means)
                avg_close_high = np.mean(topk_close_high_means)
                avg_close_low = np.mean(topk_close_low_means)
                avg_close_sharp = np.mean(topk_close_sharps)
                avg_1d_close = np.mean(topk_1d_close_means)
                avg_topk_miss = np.mean(topk_misses)
                avg_topk_ret = np.mean(topk_ret_means)
                feature_importance, feature_name = zip(*sorted(list(zip(feature_importance, feature_name)), key=lambda x:-x[0]))
                exp_result = {
                    "label": label,
                    "exp_config": exp_config,
                    "avg_epoch": avg_epoch,
                    "avg_miss": avg_topk_miss,
                    "avg_topk_ret": avg_topk_ret,
                    "avg_ap": avg_ap,
                    "avg_auc": avg_auc,
                    "avg_high": agv_high, 
                    "avg_low": avg_low,
                    "avg_sharp": avg_sharp,
                    "avg_close_open": avg_close_open,
                    "avg_close_high": avg_close_high,
                    "avg_close_low": avg_close_low,
                    "avg_close_sharp": avg_close_sharp,
                    "avg_1d_close": avg_1d_close,
                    "feature_importance": str(feature_importance[:50]),
                    "feature_name": str(feature_name[:50])
                }
                json.dump(exp_result, open(os.path.join(configured_exp_dir, "result.json"), 'w'), indent=4)
                
                agg_result["sharp_exp"]["_".join([label, exp_config])] = exp_result
                if close_open_strategy:
                    agg_result["close_open_strategy"]["_".join([label, exp_config])] = exp_result
                if close_high_strategy:
                    agg_result["close_high_strategy"]["_".join([label, exp_config])] = exp_result
                
                def compare_large(cur, best, container):
                    if cur > best:
                        best = cur
                        container = exp_result
                    return best, container
                
                max_high_rate, max_high_rate_config = compare_large(agv_high, max_high_rate, max_high_rate_config)
                max_avg_ap, max_avg_ap_config = compare_large(avg_ap, max_avg_ap, max_avg_ap_config)
                max_sharp_rate, max_sharp_rate_config = compare_large(avg_sharp, max_sharp_rate, max_sharp_rate_config)
                max_next_1d_close_2d_open_gain, max_next_1d_close_2d_open_gain_config = compare_large(avg_close_open, max_next_1d_close_2d_open_gain, max_next_1d_close_2d_open_gain_config)
                max_avg_close_high, max_avg_close_high_config = compare_large(avg_close_high, max_avg_close_high, max_avg_close_high_config)
                max_avg_close_sharp_rate, max_avg_close_sharp_rate_config = compare_large(avg_close_sharp, max_avg_close_sharp_rate, max_avg_close_sharp_rate_config)

                    
        agg_result["sharp_exp"] = to_dict(sorted(agg_result["sharp_exp"].items(), key=lambda x:-x[1]["avg_sharp"]))
        agg_result["topk_miss_exp"] = to_dict(sorted(agg_result["sharp_exp"].items(), key=lambda x:x[1]["avg_miss"]))
        agg_result["topk_ret_exp"] = to_dict(sorted(agg_result["sharp_exp"].items(), key=lambda x:-x[1]["avg_topk_ret"]))
        agg_result["sharp_1d_safe_exp"] = to_dict(sorted(agg_result["sharp_exp"].items(), key=lambda x:-x[1]["avg_sharp"]*x[1]["avg_1d_close"]))
        agg_result["high_exp"] = to_dict(sorted(agg_result["sharp_exp"].items(), key=lambda x:-x[1]["avg_high"]))
        agg_result["close_open_strategy"] = to_dict(sorted(agg_result["close_open_strategy"].items(), key=lambda x:-x[1]["avg_close_open"]))
        agg_result["close_high_strategy"] = to_dict(sorted(agg_result["close_high_strategy"].items(), key=lambda x:-x[1]["avg_close_high"]))
        agg_result["close_sharp_strategy"] = to_dict(sorted(agg_result["close_high_strategy"].items(), key=lambda x:-x[1]["avg_close_sharp"]))
        
        agg_result["best_sharp_exp"] = dict()
        agg_result["best_sharp_exp"]["ap"] = max_avg_ap_config
        agg_result["best_sharp_exp"]["sharp"] = max_sharp_rate_config
        agg_result["best_sharp_exp"]["high"] = max_high_rate_config
        agg_result["best_close_open_strategy"] = max_next_1d_close_2d_open_gain_config
        agg_result["best_close_high_strategy"] = max_avg_close_high_config
        agg_result["best_close_sharp_strategy"] = max_avg_close_sharp_rate_config
    json.dump(all_agg_result, open(os.path.join(ana_dir, "agg_info.json"), 'w'), indent=4)
            
            
            
if __name__ == "__main__":
    agg_prediction_info()
    