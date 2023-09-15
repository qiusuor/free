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

def check():
    for file in tqdm(os.listdir(DAILY_DIR)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        df = joblib.load(path)
        assert len(set(df.index)) == len(df.index), code
        if "y_next_10_d_high_ratio" in df.columns:
            assert len(df) < 240 or df["y_next_10_d_high_ratio"].max() < 3, (code, df[df["y_next_10_d_high_ratio"] >=3])
    

if __name__ == "__main__":
    check()
    
    