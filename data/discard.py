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

def discard_one(path):
    df = joblib.load(path)
    df = df[df["volume"] != 0]
    
    keep_columns = "code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST,factor,price".split(",")
    df = df[keep_columns]
    df.to_csv(path.replace(".pkl", ".csv"))
    dump(df, path)
    

def discard_labels():
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
    pool.imap_unordered(discard_one, paths)
    pool.close()
    pool.join()
     
if __name__ == "__main__":
    discard_labels()
    