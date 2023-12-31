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

def check_daily():
    last_trade_day = get_last_trade_day(update=False)
    no_last_day_data_codes = set()
    for path in main_board_stocks():
        df = joblib.load(path)
        code = os.path.basename(path)
        assert len(set(df.index)) == len(df.index), code
        if "y_next_10_d_high_ratio" in df.columns:
            assert len(df) < 240 or df["y_next_10_d_high_ratio"].max() < 3, (code, df[df["y_next_10_d_high_ratio"] >=3])
        if to_int_date(df.index[-1]) != last_trade_day:
            # print(code, "no data at last trade day!")
            no_last_day_data_codes.add(code)
    print(no_last_day_data_codes)
    return no_last_day_data_codes
    

def check_minutes(no_last_day_data_codes, check_dir):
    last_trade_day = get_last_trade_day(update=False)
    no_last_fzline_codes = set()
    for file in tqdm(os.listdir(check_dir)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(check_dir, file)
        df = joblib.load(path)
        assert len(set(df.index)) == len(df.index), code
        
        if to_int_date(df.index[-1]) != last_trade_day and code not in no_last_day_data_codes:
            no_last_fzline_codes.add(code)
        
            
            
if __name__ == "__main__":
    no_last_day_data_codes = check_daily()
    # if os.path.exists(TDX_MINUTE_DIR):
    #     no_last_fzline_codes = check_minutes(no_last_day_data_codes, TDX_MINUTE_DIR)
    # if os.path.exists(MINUTE_DIR):
    #     check_minutes(no_last_fzline_codes, MINUTE_DIR)
    
    