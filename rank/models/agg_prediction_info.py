from config import *
from utils import *
import os
import json
import numpy as np
from matplotlib import pyplot as plt

def to_dict(sorted_items):
    keys, vals = [], []
    for key, val in sorted_items:
        keys.append(key)
        vals.append(val)
    return dict(zip(keys, vals))

def plot_sharp_rate(days, sharps, img_path):
    plt.clf()
    plt.figure(figsize=(16, 8))
    plt.plot(days, sharps)
    plt.xlabel("days")
    plt.ylabel("sharps")
    plt.tick_params(axis='both', labelsize=14)
    plt.xticks(rotation=90, fontsize=14)
    plt.grid()
    plt.savefig(img_path)
        
def agg_prediction_info(ana_dir=EXP_DIR, last_n_day=TEST_N_LAST_DAY):
    all_agg_result = dict()
    for topk in [3, 5, 10]:
        agg_result = all_agg_result["Top-{}".format(topk)] = dict()
        agg_result["sharp_exp"] = dict()
        max_high_rate = -np.inf
        max_high_rate_config = None
        max_sharp_rate = -np.inf
        max_sharp_rate_config = None
        max_avg_ap = -np.inf
        max_avg_ap_config = None
        max_next_1d_close_2d_open_gain = -np.inf
        max_next_1d_close_2d_open_gain_config = None
        for label in tqdm(os.listdir(ana_dir)):
            if not label.startswith("y") and not label.startswith("dy"): continue
            label_dir = os.path.join(ana_dir, label)
            for exp_config in os.listdir(label_dir):
                configured_exp_dir = os.path.join(label_dir, exp_config)
                if not os.path.isdir(configured_exp_dir): continue
                epochs, topk_high_means, topk_low_means, topk_sharp_means, topk_gain_means, topk_1d_close_means = [], [], [], [], [], []
                topk_misses, topk_ret_means = [], []
                if len(os.listdir(configured_exp_dir)) < last_n_day: continue
                finished = True
                val_start_days = list(filter(lambda x: os.path.isdir(os.path.join(configured_exp_dir, x)), os.listdir(configured_exp_dir)))
                if len(val_start_days) > last_n_day:
                    val_start_days = list(sorted(val_start_days, key=lambda x:int(x)))[-last_n_day-2:-2]
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
                    topk_high_mean = meta["mean_val"]["top{}_watch".format(topk)]["y_next_2_d_high_ratio_topk_{}_mean".format(topk)]
                    topk_low_mean = meta["mean_val"]["top{}_watch".format(topk)]["y_next_2_d_low_ratio_topk_{}_mean".format(topk)]
                    topk_gain_mean = meta["mean_val"]["top{}_watch".format(topk)]["y_next_1d_close_2d_open_rate_topk_{}_mean".format(topk)] if "y_next_1d_close_2d_open_rate_topk_{}_mean".format(topk) in meta["mean_val"]["top{}_watch".format(topk)] else 0
                    topk_1d_close_mean = meta["mean_val"]["top{}_watch".format(topk)]["y_next_1d_close_rate_topk_{}_mean".format(topk)] if "y_next_1d_close_rate_topk_{}_mean".format(topk) in meta["mean_val"]["top{}_watch".format(topk)] else 0
                    
                    topk_miss = meta["mean_val"]["top{}_miss".format(topk)]
                    topk_ret_mean = meta["mean_val"]["top{}_watch".format(topk)]["y_next_2_d_ret_topk_{}_mean".format(topk)]
                    
                    topk_sharp_mean = topk_high_mean * topk_low_mean
                    
                    epochs.append(epoch)
                    topk_high_means.append(topk_high_mean)
                    topk_low_means.append(topk_low_mean)
                    topk_sharp_means.append(topk_sharp_mean)
                    topk_gain_means.append(topk_gain_mean)
                    topk_1d_close_means.append(topk_1d_close_mean)
                    topk_misses.append(topk_miss)
                    topk_ret_means.append(topk_ret_mean)
                    
                if not finished: continue
                avg_epoch = np.mean(epochs)
                agv_high = np.mean(topk_high_means)
                avg_low = np.mean(topk_low_means)
                avg_sharp = np.mean(topk_sharp_means)
                avg_close_open = np.mean(topk_gain_means)
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
                    "avg_high": agv_high, 
                    "avg_low": avg_low,
                    "avg_sharp": avg_sharp,
                    "avg_1d_open_close": avg_1d_close,
                    "feature_importance": str(feature_importance[:50]),
                    "feature_name": str(feature_name[:50])
                }
                json.dump(exp_result, open(os.path.join(configured_exp_dir, "result.json"), 'w'), indent=4)
                
                agg_result["sharp_exp"]["_".join([label, exp_config])] = exp_result
                
                def compare_large(cur, best, container):
                    if cur > best:
                        best = cur
                        container = exp_result
                    return best, container
                
                max_high_rate, max_high_rate_config = compare_large(agv_high, max_high_rate, max_high_rate_config)
                max_sharp_rate, max_sharp_rate_config = compare_large(avg_sharp, max_sharp_rate, max_sharp_rate_config)
                max_next_1d_close_2d_open_gain, max_next_1d_close_2d_open_gain_config = compare_large(avg_close_open, max_next_1d_close_2d_open_gain, max_next_1d_close_2d_open_gain_config)
                plot_sharp_rate(val_start_days, topk_sharp_means, os.path.join(configured_exp_dir, "sharps_{}.png".format(topk)))
                    
        agg_result["sharp_exp"] = to_dict(sorted(agg_result["sharp_exp"].items(), key=lambda x:-x[1]["avg_sharp"]))
        agg_result["topk_miss_exp"] = to_dict(sorted(agg_result["sharp_exp"].items(), key=lambda x:x[1]["avg_miss"]))
        agg_result["topk_ret_exp"] = to_dict(sorted(agg_result["sharp_exp"].items(), key=lambda x:-x[1]["avg_topk_ret"]))
        agg_result["1d_open_close_exp"] = to_dict(sorted(agg_result["sharp_exp"].items(), key=lambda x:-x[1]["avg_1d_open_close"]))
        agg_result["high_exp"] = to_dict(sorted(agg_result["sharp_exp"].items(), key=lambda x:-x[1]["avg_high"]))
        
        agg_result["best_sharp_exp"] = dict()
        agg_result["best_sharp_exp"]["sharp"] = max_sharp_rate_config
        agg_result["best_sharp_exp"]["high"] = max_high_rate_config
    json.dump(all_agg_result, open(os.path.join(ana_dir, "agg_info.json"), 'w'), indent=4)
            
            
            
if __name__ == "__main__":
    agg_prediction_info()
    