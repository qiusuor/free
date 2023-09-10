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
    ind = joblib.load(INDUSTRY_INFO)
    df["industry"] = ind[df.code[0]]
    df["industry"] = df["industry"].astype('category')
    df.to_csv(path.replace(".pkl", ".csv"))
    dump(df, path)
    

def inject_industry():
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
    pool.imap_unordered(inject_one, paths)
    pool.close()
    pool.join()
     
if __name__ == "__main__":
    inject_industry()