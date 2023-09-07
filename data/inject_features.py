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
    
    df["ma_5"] = abstract.SMA(df, timeperiod=5)
    df["ma_10"] = abstract.SMA(df, timeperiod=10)
    df["ma_30"] = abstract.SMA(df, timeperiod=30)
    df["ma_60"] = abstract.SMA(df, timeperiod=60)
    df["ma_120"] = abstract.SMA(df, timeperiod=120)
    df["ma_240"] = abstract.SMA(df, timeperiod=240)
    
    df["dema_5"] = abstract.DEMA(df, timeperiod=5)
    df["dema_10"] = abstract.DEMA(df, timeperiod=10)
    df["dema_30"] = abstract.DEMA(df, timeperiod=30)
    df["dema_60"] = abstract.DEMA(df, timeperiod=60)
    df["dema_120"] = abstract.DEMA(df, timeperiod=120)
    df["dema_240"] = abstract.DEMA(df, timeperiod=240)
    
    df["ema_5"] = abstract.EMA(df, timeperiod=5)
    df["ema_10"] = abstract.EMA(df, timeperiod=10)
    df["ema_30"] = abstract.EMA(df, timeperiod=30)
    df["ema_60"] = abstract.EMA(df, timeperiod=60)
    df["ema_120"] = abstract.EMA(df, timeperiod=120)
    df["ema_240"] = abstract.EMA(df, timeperiod=240)
    
    df["rsi_5"] = abstract.RSI(df, timeperiod=5)
    df["rsi_10"] = abstract.RSI(df, timeperiod=10)
    df["rsi_30"] = abstract.RSI(df, timeperiod=30)
    df["rsi_60"] = abstract.RSI(df, timeperiod=60)
    df["rsi_120"] = abstract.RSI(df, timeperiod=120)
    df["rsi_240"] = abstract.RSI(df, timeperiod=240)
    
    df["adx_14"] = abstract.ADX(df, timeperiod=14)
    df["natr_14"] = abstract.NATR(df, timeperiod=14)
    df["obv"] = abstract.OBV(df)
    df["type_price"] = abstract.TYPPRICE(df)
    df["avg_price"] = abstract.AVGPRICE(df)
    df["weighted_price"] = abstract.WCLPRICE(df)
    df["med_price"] = abstract.MEDPRICE(df)
    df["ht_dc_period"] = abstract.HT_DCPERIOD(df)
    
    df["ht_trend_mode"] = abstract.HT_TRENDMODE(df)
    df["beta_5"] = abstract.BETA(df, timeperiod=5)
    df["correl_30"] = abstract.CORREL(df, timeperiod=30)
    df["linear_reg_5"] = abstract.LINEARREG(df, timeperiod=5)
    df["linear_reg_10"] = abstract.LINEARREG(df, timeperiod=10)
    df["linear_reg_22"] = abstract.LINEARREG(df, timeperiod=22)
    df["linear_reg_30"] = abstract.LINEARREG(df, timeperiod=30)
    df["linear_reg_60"] = abstract.LINEARREG(df, timeperiod=60)
    df["linear_reg_120"] = abstract.LINEARREG(df, timeperiod=120)
    df["linear_reg_240"] = abstract.LINEARREG(df, timeperiod=240)
    
    df["stddev_5"] = abstract.STDDEV(df, timeperiod=5, nbdev=1.)
    df["stddev_10"] = abstract.STDDEV(df, timeperiod=10, nbdev=1.)
    df["stddev_22"] = abstract.STDDEV(df, timeperiod=22, nbdev=1.)
    df["stddev_30"] = abstract.STDDEV(df, timeperiod=30, nbdev=1.)
    df["stddev_60"] = abstract.STDDEV(df, timeperiod=60, nbdev=1.)
    df["stddev_120"] = abstract.STDDEV(df, timeperiod=120, nbdev=1.)
    df["stddev_240"] = abstract.STDDEV(df, timeperiod=240, nbdev=1.)
    
    df["tsf_5"] = abstract.TSF(df, timeperiod=5)
    df["tsf_10"] = abstract.TSF(df, timeperiod=10)
    df["tsf_22"] = abstract.TSF(df, timeperiod=22)
    df["tsf_30"] = abstract.TSF(df, timeperiod=30)
    df["tsf_60"] = abstract.TSF(df, timeperiod=60)
    df["tsf_120"] = abstract.TSF(df, timeperiod=120)
    df["tsf_240"] = abstract.TSF(df, timeperiod=240)
    
    df["var_5"] = abstract.VAR(df, timeperiod=5, nbdev=1.)
    df["var_10"] = abstract.VAR(df, timeperiod=10, nbdev=1.)
    df["var_22"] = abstract.VAR(df, timeperiod=22, nbdev=1.)
    df["var_30"] = abstract.VAR(df, timeperiod=30, nbdev=1.)
    df["var_60"] = abstract.VAR(df, timeperiod=60, nbdev=1.)
    df["var_120"] = abstract.VAR(df, timeperiod=120, nbdev=1.)
    df["var_240"] = abstract.VAR(df, timeperiod=240, nbdev=1.)
    
    
    max_min_period = [3, 5, 10, 30, 60, 120, 240]
    max_name = ["close", "high"]
    min_name = ["close", "low"]
    
    for period in max_min_period:
        for name in max_name:
            df["max_{}_{}".format(name, period)] = df[name].rolling(period).max()
        for name in min_name:
            df["min_{}_{}".format(name, period)] = df[name].rolling(period).min()
            
    chip_div_avg_period = [3, 5, 10, 30, 60, 120, 240]
    
    def calc_chip_avg_wrapper(df):
        def calc_chip_avg(df_w):
            dfi =  df[(df.index >= df_w.index[0]) & (df.index <= df_w.index[-1])]
            avg = sum(dfi["amount"]) / sum(dfi["volume"])
            return avg
        return calc_chip_avg
    
    def calc_chip_div_wrapper(df):
        def calc_chip_div(df_w):
            dfi =  df[(df.index >= df_w.index[0]) & (df.index <= df_w.index[-1])]
            avg = sum(dfi["amount"]) / sum(dfi["volume"])
            sum_vol = sum(dfi["volume"])
            div = sum((1 - dfi["price"] / avg) ** 2 * dfi["volume"] / sum_vol)
            return div * 1000
        return calc_chip_div
    
    for period in chip_div_avg_period:
        df["chip_avg_{}".format(period)] = df["close"].rolling(period).apply(calc_chip_avg_wrapper(df), raw=False)
        df["chip_div_{}".format(period)] = df["close"].rolling(period).apply(calc_chip_div_wrapper(df), raw=False)
    
    for metric in talib.get_function_groups()["Pattern Recognition"]:
        df[metric] = abstract.Function(metric)(df)

    
    if "inphase" not in df.columns:
        df = pd.concat([df, abstract.HT_PHASOR(df)], axis=1)
        
    if "upperband" not in df.columns:
        df = pd.concat([df, abstract.BBANDS(df, timeperiod=5, nbdevup=2., nbdevdn=2., matype=0)], axis=1)
    
    if "macd" not in df.columns:
        df = pd.concat([df, abstract.MACD(df, fastperiod=10, slowperiod=22, signalperiod=5)], axis=1)
       
    df.to_csv(path.replace(".pkl", ".csv"))
    # print(df)
    # print(list(df.columns))
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
    # inject_one(paths[0])
    pool.imap_unordered(inject_one, paths)
    pool.close()
    pool.join()
     
if __name__ == "__main__":
    # print(talib.get_function_groups())
    inject()