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
import shutil
from data.fetch_core import dealTime

warnings.filterwarnings("ignore")

def discard_one(path):
    df = pd.read_csv(path)
    dealTime(df)
    inject_industry_and_name(df)
    
    if "code_name" not in df.columns or not isinstance(df.code_name[-1], str) or "ST" in df.code_name[-1] or "st" in df.code_name[-1] or "sT" in df.code_name[-1]:
        return
    
    keep_columns = "code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST,factor,price,value".split(",")
    df = df[keep_columns]
    df = df[:to_date(SEARCH_END_DAY)]
    path = os.path.join(DAILY_DIR, os.path.basename(path))
    df.to_csv(path)
    dump(df, path.replace(".csv", ".pkl"))
    

def discard_info():
    pool = Pool(THREAD_NUM)
    paths = []
    src_dir = DAILY_DOWLOAD_DIR
    remove_dir(DAILY_DIR)
    make_dir(DAILY_DIR)
    for file in tqdm(os.listdir(src_dir)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".csv"):
            continue
        path = os.path.join(src_dir, file)
        paths.append(path)
    # discard_one(paths[0])
    # exit(0)
    pool.imap_unordered(discard_one, paths)
    pool.close()
    pool.join()
     
if __name__ == "__main__":
    discard_info()
    