import numpy as np
import pandas as pd
from talib import abstract
import talib
from multiprocessing import Pool
from config import *
from utils import *
from tqdm import tqdm
from joblib import dump
import warnings
import platform

warnings.filterwarnings("ignore")

def inject_one(path):
    df = joblib.load(path)
    
    df["y_open_close"] = df["open_close"].shift(-1)
    df["y_next_1d_close_rate"] = df["close"].shift(-1) / df["open"].shift(-1)
    df["y_next_1d_close_rate"].fillna(0, inplace=True)
    df["y_next_1d_close_rate_01"] = df["y_next_1d_close_rate"] >= 1.01
    df["y_next_1d_close_rate_02"] = df["y_next_1d_close_rate"] >= 1.02
    df["y_next_1d_close_rate_03"] = df["y_next_1d_close_rate"] >= 1.03
    df["y_next_1d_close_rate_04"] = df["y_next_1d_close_rate"] >= 1.04
    df["y_next_1d_close_rate_05"] = df["y_next_1d_close_rate"] >= 1.05
    df["y_next_1d_close_rate_08"] = df["y_next_1d_close_rate"] >= 1.08
    df["y_next_1d_close_rate_095"] = df["y_next_1d_close_rate"] >= 1.095
    df["y_next_1d_ret"] = df["open"].shift(-1) / df["close"]
    df["y_next_1d_ret_close"] = df["close"].shift(-1) / df["close"]
    df["y_next_1d_up_to_limit"] =  is_limit_up(df).shift(-1).astype(bool)
    df["y_next_1d_up_to_limit"].fillna(False, inplace=True)
    
    df["y_rank_1d_label"] = df["y_next_1d_up_to_limit"].astype(int)
    df["y_rank_1d_label"][(df["y_next_1d_close_rate"] > 1.07) & df["y_next_1d_up_to_limit"]] = 4
    df["y_rank_1d_label"][(df["y_next_1d_close_rate"] > 1.05) & df["y_next_1d_up_to_limit"]] = 3
    df["y_rank_1d_label"][(df["y_next_1d_close_rate"] > 1.02) & df["y_next_1d_up_to_limit"]] = 2
    
    df = data_filter(df)
    df.to_csv(path.replace(".pkl", ".csv"))
    dump(df, path)
    

def inject_labels():
    pool = Pool(THREAD_NUM)
    paths = main_board_stocks()
    # inject_one(paths[0])
    pool.imap_unordered(inject_one, paths)
    pool.close()
    pool.join()
     
if __name__ == "__main__":
    inject_labels()
    