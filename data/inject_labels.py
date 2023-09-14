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
from data.inject_joint_label import injecto_joint_label

warnings.filterwarnings("ignore")

def inject_one(path):
    df = joblib.load(path)
    df = df[df["volume"] != 0]
    
    date = df.index
    close = df.close.values
    high = df.high.values
    open = df.open.values
    low = df.low.values
    price = df.price.values
    volume = df.volume.values
    amount = df.amount.values
    turn = df.turn.values
    
    for label in labels:
        up, nday, ratio = explain_label(label=label)
        df[label] = get_labels(open, close, high, low, price, turn, hold_day=nday, expect_gain=ratio, tolerent_pay=ratio, up=up)
        
    future_n_day_high_low = [2, 3, 5, 10, 22]
    
    for n_day in future_n_day_high_low:
        df["y_next_{}_d_ret".format(n_day)] = df["close"].shift(-n_day) / df["open"].shift(-1)
        df["y_next_{}_d_high".format(n_day)] = df["high"].rolling(n_day).apply(lambda x:max(x[1:])).shift(-n_day)
        df["y_next_{}_d_high_ratio".format(n_day)] = df["y_next_{}_d_high".format(n_day)] / df["open"].shift(-1)
        df["y_next_{}_d_low".format(n_day)] = df["low"].rolling(n_day).apply(lambda x:min(x[1:])).shift(-n_day)
        df["y_next_{}_d_low_ratio".format(n_day)] = df["y_next_{}_d_low".format(n_day)] / df["open"].shift(-1)
        
    df.to_csv(path.replace(".pkl", ".csv"))
    dump(df, path)
    

def inject_labels():
    pool = Pool(THREAD_NUM)
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
    injecto_joint_label()
     
if __name__ == "__main__":
    inject_labels()
    