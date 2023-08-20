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
    df.to_csv(path.replace(".pkl", ".csv"))
    dump(df, path)
    

def inject():
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
    inject_one(paths[0])
    pool.imap_unordered(inject_one, paths)
    pool.close()
    pool.join()
     
if __name__ == "__main__":
    # print(talib.get_function_groups())
    inject()