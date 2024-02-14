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

def inject_ta_features(df):
    df["rsi_5"] = abstract.RSI(df, timeperiod=5)
    df["rsi_10"] = abstract.RSI(df, timeperiod=10)
    df["rsi_30"] = abstract.RSI(df, timeperiod=30)
    df["rsi_60"] = abstract.RSI(df, timeperiod=60)
    df["rsi_120"] = abstract.RSI(df, timeperiod=120)
    df["rsi_240"] = abstract.RSI(df, timeperiod=240)
    
    df["adx_2"] = abstract.ADX(df, timeperiod=2)
    df["adx_5"] = abstract.ADX(df, timeperiod=5)
    df["adx_7"] = abstract.ADX(df, timeperiod=7)
    df["adx_14"] = abstract.ADX(df, timeperiod=14)
    df["adx_30"] = abstract.ADX(df, timeperiod=30)
    df["natr_2"] = abstract.NATR(df, timeperiod=2)
    df["natr_3"] = abstract.NATR(df, timeperiod=3)
    df["natr_5"] = abstract.NATR(df, timeperiod=5)
    df["natr_10"] = abstract.NATR(df, timeperiod=10)
    df["natr_14"] = abstract.NATR(df, timeperiod=14)
    df["natr_22"] = abstract.NATR(df, timeperiod=22)
    df["natr_60"] = abstract.NATR(df, timeperiod=60)
    df["natr_120"] = abstract.NATR(df, timeperiod=120)
    df["natr_240"] = abstract.NATR(df, timeperiod=240)

    df["beta_2"] = abstract.BETA(df, timeperiod=2)
    df["beta_3"] = abstract.BETA(df, timeperiod=3)
    df["beta_5"] = abstract.BETA(df, timeperiod=5)
    df["beta_7"] = abstract.BETA(df, timeperiod=7)
    df["beta_10"] = abstract.BETA(df, timeperiod=10)
    df["beta_22"] = abstract.BETA(df, timeperiod=22)
    df["beta_30"] = abstract.BETA(df, timeperiod=30)
    df["beta_60"] = abstract.BETA(df, timeperiod=60)
    df["beta_120"] = abstract.BETA(df, timeperiod=120)
    df["beta_240"] = abstract.BETA(df, timeperiod=240)
    df["correl_2"] = abstract.CORREL(df, timeperiod=2)
    df["correl_3"] = abstract.CORREL(df, timeperiod=3)
    df["correl_5"] = abstract.CORREL(df, timeperiod=5)
    df["correl_10"] = abstract.CORREL(df, timeperiod=10)
    df["correl_22"] = abstract.CORREL(df, timeperiod=22)
    df["correl_30"] = abstract.CORREL(df, timeperiod=30)
    df["correl_60"] = abstract.CORREL(df, timeperiod=60)
    df["correl_120"] = abstract.CORREL(df, timeperiod=120)
    df["correl_240"] = abstract.CORREL(df, timeperiod=240)

    df["stddev_5"] = abstract.STDDEV(df, timeperiod=5, nbdev=1.)
    df["stddev_10"] = abstract.STDDEV(df, timeperiod=10, nbdev=1.)
    df["stddev_22"] = abstract.STDDEV(df, timeperiod=22, nbdev=1.)
    df["stddev_30"] = abstract.STDDEV(df, timeperiod=30, nbdev=1.)
    df["stddev_60"] = abstract.STDDEV(df, timeperiod=60, nbdev=1.)
    df["stddev_120"] = abstract.STDDEV(df, timeperiod=120, nbdev=1.)
    df["stddev_240"] = abstract.STDDEV(df, timeperiod=240, nbdev=1.)
    
    df["var_5"] = abstract.VAR(df, timeperiod=5, nbdev=1.)
    df["var_10"] = abstract.VAR(df, timeperiod=10, nbdev=1.)
    df["var_22"] = abstract.VAR(df, timeperiod=22, nbdev=1.)
    df["var_30"] = abstract.VAR(df, timeperiod=30, nbdev=1.)
    df["var_60"] = abstract.VAR(df, timeperiod=60, nbdev=1.)
    df["var_120"] = abstract.VAR(df, timeperiod=120, nbdev=1.)
    df["var_240"] = abstract.VAR(df, timeperiod=240, nbdev=1.)
    
    for metric in talib.get_function_groups()["Pattern Recognition"]:
        df[metric] = abstract.Function(metric)(df)

    
    if "inphase" not in df.columns:
        df = pd.concat([df, abstract.HT_PHASOR(df)], axis=1)
        
    if "upperband" not in df.columns:
        df = pd.concat([df, abstract.BBANDS(df, timeperiod=5, nbdevup=2., nbdevdn=2., matype=0)], axis=1)
    
    if "macd" not in df.columns:
        df = pd.concat([df, abstract.MACD(df, fastperiod=10, slowperiod=22, signalperiod=5)], axis=1)
    return df
    
def inject_chip_features(df):
    chip_div_avg_period = [3, 5, 10, 30, 60, 120, 240]
    
    @pandas_rolling_agg(df)
    def calc_chip_avg(dfi):
        avg = sum(dfi["amount"]) / sum(dfi["volume"])
        return avg
    
    @pandas_rolling_agg(df)
    def calc_chip_div(dfi):
        avg = sum(dfi["amount"]) / sum(dfi["volume"])
        sum_vol = sum(dfi["volume"])
        div = sum((1 - dfi["price"] / avg) ** 2 * dfi["volume"] / sum_vol)
        return div * 1000
    
    for period in chip_div_avg_period:
        df["chip_avg_{}".format(period)] = df["close"].rolling(period).apply(calc_chip_avg, raw=False)
        df["chip_div_{}".format(period)] = df["close"].rolling(period).apply(calc_chip_div, raw=False)
        df["price_div_chip_avg_{}".format(period)] = df["price"] / (df["chip_avg_{}".format(period)] + 1e-9)
    return df
    
def inject_price_turn_features(df):
    max_min_period = [3, 5, 10, 30, 60, 120, 240]
    max_name = ["close", "high"]
    min_name = ["close", "low"]
    
    for period in max_min_period:
        df["mean_price_{}".format(period)] = df["price"].rolling(period).mean()
        for name in max_name:
            df["max_{}_{}".format(name, period)] = df[name].rolling(period).max() / df[name]
        for name in min_name:
            df["min_{}_{}".format(name, period)] = df[name].rolling(period).min() / df[name]
    
    pct_period = [3, 5, 10, 30, 60, 120, 240]
    
    def calc_rank_pct(df_w):
        return df_w.rank(pct=True)[-1]
    
    for period in pct_period:
         df["pct_close_{}".format(period)] = df["close"].rolling(period).apply(calc_rank_pct, raw=False)
         df["pct_low_{}".format(period)] = df["low"].rolling(period).apply(calc_rank_pct, raw=False)
         df["pct_price_{}".format(period)] = df["price"].rolling(period).apply(calc_rank_pct, raw=False)
    
    turn_period = [3, 5, 10, 30, 60, 120, 240]
    for period in turn_period:
        df["mean_turn_{}".format(period)] = df["turn"].rolling(period).mean()
        df["max_turn_{}".format(period)] = df["turn"].rolling(period).max()
        df["min_turn_{}".format(period)] = df["turn"].rolling(period).min()
        df["std_turn_{}".format(period)] = df["turn"].rolling(period).std()
        df["turn_div_mean_turn_{}".format(period)] = df["turn"] / (df["mean_turn_{}".format(period)] + 1e-9)
    return df
    
def inject_alpha_features(df):
    
    @pandas_rolling_agg(df)
    def alpha_001(df_w):
        """
            (-1 * CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            window = 7
        """
        rank1 = df_w["volume"].apply(np.log).rolling(2).apply(lambda x:x[1]-x[0])[1:].rank()
        rank2 = ((df_w["close"] - df_w["open"]) / df_w["open"])[1:].rank()
        return -rank1.corr(rank2)
    df["alpha_001"] = df["close"].rolling(7).apply(alpha_001, raw=False)
    
    
    def alpha_002(df):
        """
            -1 * delta((((close-low)-(high-close))/((high-low)),1))
            window = 2
        """
        v1 = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
        return -v1.rolling(2).apply(lambda x:x[1]-x[0])
    df["alpha_002"] = alpha_002(df)
        
    
    return df

    
def inject_style_feature(df):
    df["up_shadow"] = (df["high"] - df["close"]) / (df["close"] + 1e-6)
    df["down_shadow"] = (df["low"] - df["close"]) / (df["close"] + 1e-6)
    df["limit_up"] = is_limit_up(df)
    df["reach_limit_up"] = is_reach_limit_up(df)
    df["limit_up_1d"] = is_limit_up(df)
    df["limit_up_2d"] = is_limit_up(df) & (df["limit_up_1d"].shift(1))
    df["limit_up_3d"] = is_limit_up(df) & (df["limit_up_2d"].shift(1))
    df["limit_up_4d"] = is_limit_up(df) & (df["limit_up_3d"].shift(1))
    df["limit_up_5d"] = is_limit_up(df) & (df["limit_up_4d"].shift(1))
    df["limit_up_6d"] = is_limit_up(df) & (df["limit_up_5d"].shift(1))
    df["limit_up_7d"] = is_limit_up(df) & (df["limit_up_6d"].shift(1))
    df["limit_up_8d"] = is_limit_up(df) & (df["limit_up_7d"].shift(1))
    df["limit_up_9d"] = is_limit_up(df) & (df["limit_up_8d"].shift(1))
    df["limit_up_9d_plus"] = is_limit_up(df) & (df["limit_up_9d"].shift(1))
    
    df["limit_up_1d"] = df["limit_up_1d"] & (~df["limit_up_2d"])
    df["limit_up_2d"] = df["limit_up_2d"] & (~df["limit_up_3d"])
    df["limit_up_3d"] = df["limit_up_3d"] & (~df["limit_up_4d"])
    df["limit_up_4d"] = df["limit_up_4d"] & (~df["limit_up_5d"])
    df["limit_up_5d"] = df["limit_up_5d"] & (~df["limit_up_6d"])
    df["limit_up_6d"] = df["limit_up_6d"] & (~df["limit_up_7d"])
    df["limit_up_7d"] = df["limit_up_7d"] & (~df["limit_up_8d"])
    df["limit_up_8d"] = df["limit_up_8d"] & (~df["limit_up_9d"])
    df["limit_up_9d"] = df["limit_up_9d"] & (~df["limit_up_9d_plus"])
    
    df["limit_down"] = is_limit_down(df)
    df["reach_limit_down"] = is_reach_limit_down(df)
    df["limit_down_1d"] = is_limit_down(df)
    df["limit_down_2d"] = is_limit_down(df) & (df["limit_down_1d"].shift(1))
    df["limit_down_3d"] = is_limit_down(df) & (df["limit_down_2d"].shift(1))
    df["limit_down_4d"] = is_limit_down(df) & (df["limit_down_3d"].shift(1))
    df["limit_down_5d"] = is_limit_down(df) & (df["limit_down_4d"].shift(1))
    df["limit_down_5d_plus"] = is_limit_down(df) & (df["limit_down_5d"].shift(1))
    df["limit_down_1d"]  = df["limit_down_1d"] & (~df["limit_down_2d"])
    df["limit_down_2d"]  = df["limit_down_2d"] & (~df["limit_down_3d"])
    df["limit_down_3d"]  = df["limit_down_3d"] & (~df["limit_down_4d"])
    df["limit_down_4d"]  = df["limit_down_4d"] & (~df["limit_down_5d"])
    df["limit_down_5d"]  = df["limit_down_5d"] & (~df["limit_down_5d_plus"])
     
    df["limit_up_line"] = is_limit_up_line(df)
    df["limit_up_line_1d"] = is_limit_up_line(df) & df["limit_up_1d"]
    df["limit_up_line_2d"] = is_limit_up_line(df) & df["limit_up_2d"]
    df["limit_up_line_3d"] = is_limit_up_line(df) & df["limit_up_3d"]
    df["limit_up_line_4d"] = is_limit_up_line(df) & df["limit_up_4d"]
    df["limit_up_line_5d"] = is_limit_up_line(df) & df["limit_up_5d"]
    df["limit_up_line_6d"] = is_limit_up_line(df) & df["limit_up_6d"]
    df["limit_up_line_7d"] = is_limit_up_line(df) & df["limit_up_7d"]
    
    df["limit_down_line"] = is_limit_down_line(df)
    df["limit_down_line_1d"] = is_limit_down_line(df) & df["limit_down_1d"]
    df["limit_down_line_2d"] = is_limit_down_line(df) & df["limit_down_2d"]
    df["limit_down_line_3d"] = is_limit_down_line(df) & df["limit_down_3d"]
    df["limit_down_line_4d"] = is_limit_down_line(df) & df["limit_down_4d"]
    
def inject_one(path):
    df = joblib.load(path)
    
    inject_ta_features(df)
    if "Linux" in platform.platform():
        inject_chip_features(df)
        inject_price_turn_features(df)
        inject_alpha_features(df)
    inject_style_feature(df)
    
    # minu_feat_path = os.path.join(MINUTE_FEAT, os.path.basename(path).replace("_d_2", "_1_3"))
    # minu_feat = joblib.load(minu_feat_path)
    # minu_feat = minu_feat.set_index("date")
    # df = df.join(minu_feat, how="left")

    df.to_csv(path.replace(".pkl", ".csv"))
    dump(df, path)
    

def inject_features():
    pool = Pool(THREAD_NUM)
    paths = main_board_stocks()
    # inject_one(paths[0])
    # exit(0)
    pool.imap_unordered(inject_one, paths)
    pool.close()
    pool.join()
     
if __name__ == "__main__":
    inject_features()