from config import *
from utils import *
import os
import json
import numpy as np



def agg():
    agg_result = dict()
    agg_result["all_sharp_exp"] = dict()
    agg_result["y_next_1d_close_2d_open_rate_rank"] = dict()
    max_high_rate = -np.inf
    max_high_rate_config = None
    max_sharp_rate = -np.inf
    max_sharp_rate_config = None
    max_avg_ap = -np.inf
    max_avg_ap_config = None
    max_next_1d_close_2d_open_gain = -np.inf
    max_next_1d_close_2d_open_gain_config = None
    for label in tqdm(os.listdir(EXP_DIR)):
        calc_next_1d_close_2d_open = "y_next_1d_close_2d_open_rate_rank" in label
        if not label.startswith("y") and not label.startswith("dy"): continue
        label_dir = os.path.join(EXP_DIR, label)
        for exp_config in  os.listdir(label_dir):
            configured_exp_dir = os.path.join(label_dir, exp_config)
            if not os.path.isdir(configured_exp_dir): continue
            epochs, last_aps, last_aucs, top3_high_means, top3_low_means, top3_sharps, top3_gains = [], [], [], [], [], [], []
            if len(os.listdir(configured_exp_dir)) < 10: continue
            finished = True
            for val_start_day in os.listdir(configured_exp_dir):
                val_start_day_dir = os.path.join(configured_exp_dir, val_start_day)
                # print(val_start_day_dir)
                
                if not os.path.isdir(val_start_day_dir): continue
                if not os.path.exists(os.path.join(val_start_day_dir, "meta.json")):
                    finished = False
                    break
                meta = json.load(open(os.path.join(val_start_day_dir, "meta.json")))
                epoch = meta["info"]["epoch"]
                last_ap = meta["last_val"]["ap"]
                last_auc = meta["last_val"]["auc"]
                top3_high_mean = meta["last_val"]["top3_watch"]["y_next_2_d_high_ratio_topk_3_mean"]
                top3_low_mean = meta["last_val"]["top3_watch"]["y_next_2_d_low_ratio_topk_3_mean"]
                top3_gain = meta["last_val"]["top3_watch"]["y_next_1d_close_2d_open_rate_topk_3_mean"] if calc_next_1d_close_2d_open else 0
                top3_sharp = top3_high_mean * top3_low_mean
                epochs.append(epoch)
                last_aps.append(last_ap)
                last_aucs.append(last_auc)
                top3_high_means.append(top3_high_mean)
                top3_low_means.append(top3_low_mean)
                top3_sharps.append(top3_sharp)
                top3_gains.append(top3_gain)
                
            if not finished: continue
            avg_epoch = np.mean(epochs)
            avg_auc = np.mean(last_aucs)
            avg_ap = np.mean(last_aps)
            agv_high = np.mean(top3_high_means)
            avg_low = np.mean(top3_low_means)
            avg_sharp = np.mean(top3_sharps)
            avg_gain = np.mean(top3_gains)

            target_field = "y_next_1d_close_2d_open_rate_rank" if calc_next_1d_close_2d_open else "all_sharp_exp"
            
            agg_result[target_field][exp_config] = dict()
            agg_result[target_field][exp_config]["label"] = label
            agg_result[target_field][exp_config]["avg_epoch"] = avg_epoch
            agg_result[target_field][exp_config]["avg_ap"] = avg_ap
            agg_result[target_field][exp_config]["avg_auc"] = avg_auc
            agg_result[target_field][exp_config]["agv_high"] = agv_high
            agg_result[target_field][exp_config]["avg_low"] = avg_low
            agg_result[target_field][exp_config]["avg_sharp"] = avg_sharp
            agg_result[target_field][exp_config]["avg_gain"] = avg_gain
    
            
            if agv_high > max_high_rate:
                max_high_rate = agv_high
                max_high_rate_config = {
                    "label": label,
                    "exp_config": exp_config,
                    "avg_epoch": avg_epoch,
                    "avg_ap": avg_ap,
                    "avg_auc": avg_auc,
                    "avg_high": agv_high,
                    "avg_low": avg_low,
                    "avg_sharp": avg_sharp,
                    "avg_gain": avg_gain
                }
            if avg_ap > max_avg_ap:
                max_avg_ap = avg_ap
                max_avg_ap_config = {
                    "label": label,
                    "exp_config": exp_config,
                    "avg_epoch": avg_epoch,
                    "avg_ap": avg_ap,
                    "avg_auc": avg_auc,
                    "avg_high": agv_high,
                    "avg_low": avg_low,
                    "avg_sharp": avg_sharp,
                    "avg_gain": avg_gain
                    
                }
            if avg_sharp > max_sharp_rate:
                max_sharp_rate = avg_sharp
                max_sharp_rate_config = {
                    "label": label,
                    "exp_config": exp_config,
                    "avg_epoch": avg_epoch,
                    "avg_ap": avg_ap,
                    "avg_auc": avg_auc,
                    "avg_high": agv_high,
                    "avg_low": avg_low,
                    "avg_sharp": avg_sharp,
                    "avg_gain": avg_gain
                    
                }
            if avg_gain > max_next_1d_close_2d_open_gain:
                max_next_1d_close_2d_open_gain = avg_gain
                max_next_1d_close_2d_open_gain_config = {
                    "label": label,
                    "exp_config": exp_config,
                    "avg_epoch": avg_epoch,
                    "avg_ap": avg_ap,
                    "avg_auc": avg_auc,
                    "avg_high": agv_high,
                    "avg_low": avg_low,
                    "avg_sharp": avg_sharp,
                    "avg_gain": avg_gain
                    
                }
                
    agg_result["all_sharp_exp"] = sorted(agg_result["all_sharp_exp"].items(), key=lambda x:x[1]["avg_sharp"])
    agg_result["y_next_1d_close_2d_open_rate_rank"] = sorted(agg_result["y_next_1d_close_2d_open_rate_rank"].items(), key=lambda x:x[1]["avg_gain"])
    
    agg_result["best_sharp_exp"] = dict()
    agg_result["best_sharp_exp"]["ap"] = max_avg_ap_config
    agg_result["best_sharp_exp"]["sharp"] = max_sharp_rate_config
    agg_result["best_sharp_exp"]["high"] = max_high_rate_config
    agg_result["best_y_next_1d_close_2d_open_rate_rank"] = max_next_1d_close_2d_open_gain_config
    json.dump(agg_result, open(os.path.join(EXP_DIR, "agg_info.json"), 'w'), indent=4)
            
            
            
if __name__ == "__main__":
    agg()
    