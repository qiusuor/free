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
from data.inject_features import inject_industry_and_name

warnings.filterwarnings("ignore")

def inject_one(path):
    df = joblib.load(path)
    df = df[df["volume"] != 0]
    
    future_n_day_high_low = [2]
    df["y_next_1d_close_rate"] = df["close"].shift(-1) / df["open"].shift(-1)
    df["y_next_1d_close_rate"].fillna(0, inplace=True)
    df["y_next_1d_ret"] = df["close"].shift(-1) / df["close"]
    df["y_next_1d_up_to_limit"] =  is_limit_up(df).shift(-1)
    df["y_next_1d_up_to_limit"].fillna(0, inplace=True)
    df["y_next_1d_close_2d_open_rate"] = df["open"].shift(-2) / df["close"].shift(-1)
    for n_day in future_n_day_high_low:
        df["y_next_{}_d_ret".format(n_day)] = df["close"].shift(-n_day) / df["open"].shift(-1)
        df["y_next_{}_d_ret_03".format(n_day)] = df["y_next_{}_d_ret".format(n_day)] > 1.03
        df["y_next_{}_d_ret_04".format(n_day)] = df["y_next_{}_d_ret".format(n_day)] > 1.04
        df["y_next_{}_d_ret_05".format(n_day)] = df["y_next_{}_d_ret".format(n_day)] > 1.05
        df["y_next_{}_d_ret_07".format(n_day)] = df["y_next_{}_d_ret".format(n_day)] > 1.07
        df["y_next_{}_d_ret_095".format(n_day)] = df["y_next_{}_d_ret".format(n_day)] >= 1.095
        df["y_next_{}_d_close_high_ratio".format(n_day)] = df["high"].shift(-n_day) / df["close"].shift(-1)
        df["y_next_{}_d_close_low_ratio".format(n_day)] = df["low"].shift(-n_day) / df["close"].shift(-1)
        df["y_next_{}_d_high".format(n_day)] = df["high"].rolling(n_day).apply(lambda x:max(x[1:])).shift(-n_day)
        df["y_next_{}_d_high_ratio".format(n_day)] = df["y_next_{}_d_high".format(n_day)] / df["open"].shift(-1)
        df["y_next_{}_d_low".format(n_day)] = df["low"].rolling(n_day).apply(lambda x:min(x[1:])).shift(-n_day)
        df["y_next_{}_d_low_ratio".format(n_day)] = df["y_next_{}_d_low".format(n_day)] / df["open"].shift(-1)
        
    inject_industry_and_name(df)
    df.to_csv(path.replace(".pkl", ".csv"))
    dump(df, path)
    

def inject_labels():
    pool = Pool(8)
    paths = []
    for file in tqdm(os.listdir(DAILY_DIR)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        paths.append(path)
    # inject_one(paths[0])
    pool.imap_unordered(inject_one, paths)
    pool.close()
    pool.join()
     
if __name__ == "__main__":
    inject_labels()
    