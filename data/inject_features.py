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

def inject_ta_features(df):
    df["ma_3"] = abstract.SMA(df, timeperiod=3)
    df["ma_5"] = abstract.SMA(df, timeperiod=5)
    df["ma_6"] = abstract.SMA(df, timeperiod=6)
    df["ma_10"] = abstract.SMA(df, timeperiod=10)
    df["ma_12"] = abstract.SMA(df, timeperiod=12)
    df["ma_24"] = abstract.SMA(df, timeperiod=24)
    df["ma_30"] = abstract.SMA(df, timeperiod=30)
    df["ma_60"] = abstract.SMA(df, timeperiod=60)
    df["ma_120"] = abstract.SMA(df, timeperiod=120)
    df["ma_240"] = abstract.SMA(df, timeperiod=240)
    df["ma_bbi"] = (df["ma_3"] + df["ma_6"] + df["ma_12"] + df["ma_24"]) / 4
    
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
    
    df["adx_2"] = abstract.ADX(df, timeperiod=2)
    df["adx_5"] = abstract.ADX(df, timeperiod=5)
    df["adx_7"] = abstract.ADX(df, timeperiod=7)
    df["adx_14"] = abstract.ADX(df, timeperiod=14)
    df["adx_30"] = abstract.ADX(df, timeperiod=30)
    df["natr_5"] = abstract.NATR(df, timeperiod=5)
    df["natr_14"] = abstract.NATR(df, timeperiod=14)
    df["obv"] = abstract.OBV(df)
    df["type_price"] = abstract.TYPPRICE(df)
    df["avg_price"] = abstract.AVGPRICE(df)
    df["weighted_price"] = abstract.WCLPRICE(df)
    df["med_price"] = abstract.MEDPRICE(df)
    df["ht_dc_period"] = abstract.HT_DCPERIOD(df)
    
    df["ht_trend_mode"] = abstract.HT_TRENDMODE(df)
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
    return df
    
def inject_price_turn_features(df):
    max_min_period = [3, 5, 10, 30, 60, 120, 240]
    max_name = ["close", "high"]
    min_name = ["close", "low"]
    
    for period in max_min_period:
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

def inject_industry_and_name(df):
    ind = joblib.load(INDUSTRY_INFO)
    df["industry"] = ind[df.code[-1]]["industry"]
    df["industry"] = df["industry"].astype('category')
    df["code_name"] = ind[df.code[-1]]["code_name"]
    
    return df

def inject_shibor(df):
    shibor = joblib.load(SHIBOR_INFO)
    df["shiborON"] = shibor["shiborON"]
    df["shibor1W"] = shibor["shibor1W"]
    df["shibor2W"] = shibor["shibor2W"]
    df["shibor3M"] = shibor["shibor3M"]
    df["shibor9M"] = shibor["shibor9M"]
    df["shibor1Y"] = shibor["shibor1Y"]
    
def inject_one(path):
    df = joblib.load(path)
    df = df[df["volume"] != 0]
    
    inject_ta_features(df)
    inject_chip_features(df)
    inject_price_turn_features(df)
    inject_alpha_features(df)
    inject_industry_and_name(df)
    

    df.to_csv(path.replace(".pkl", ".csv"))
    dump(df, path)
    

def inject_features():
    # get_shibor()
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
    inject_features()