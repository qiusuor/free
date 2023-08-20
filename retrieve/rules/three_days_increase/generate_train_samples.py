from config import *
import pandas as pd
from multiprocessing import Pool
import joblib
import bisect
import os
from utils import *
import _pickle as cPickle
from tqdm import tqdm

seq_len = 5
hold_day = 2
expect_gain = 1.08
tolerent_pay = 0.97
train_val_split = 0.7
assert seq_len >= 3
assert hold_day >= 2

train_features =['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM', 'isST', 'factor', 'price','ma_5', 'ma_10', 'ma_30', 'ma_60', 'ma_120', 'ma_240', 'dema_5', 'dema_10', 'dema_30', 'dema_60', 'dema_120', 'dema_240', 'ema_5', 'ema_10', 'ema_30', 'ema_60', 'ema_120', 'ema_240', 'rsi_14', 'adx_14', 'natr_14', 'obv', 'type_price', 'avg_price', 'weighted_price', 'med_price', 'ht_dc_period', 'ht_trend_mode', 'beta_5', 'correl_30', 'linear_reg_5', 'linear_reg_10', 'linear_reg_22', 'stddev_5', 'stddev_10', 'stddev_22', 'tsf_5', 'tsf_10', 'tsf_22', 'var_5', 'var_10', 'var_22', 'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS', 'inphase', 'quadrature', 'upperband', 'middleband', 'lowerband', 'macd', 'macdsignal', 'macdhist']
train_label = "y_02_105"


def three_days_increase_retr(i, open, close, high, low, price, turn):
    if not (close[i] > close[i-1] > close[i-2]): return False
    if not (turn[i] > turn[i-1] > turn[i-2]): return False
    if turn[i] / turn[i-2] >= 2: return False
    if turn[i] > 15: return False
    if close[i] / close[i-i] >1.08: return False
    return True
    
    
# def get_label(i, open, close, high, low, price, turn):
#     for j in range(i+2, i+hold_day+1):
#         if high[j] / open[i+1] > expect_gain: return 2
#         if low[j] / open[i+1] < tolerent_pay: return 0
#     return 1

def generate():
    trade_days, last_trade_day = get_last_update_date()
    
    train_data_x = []
    val_data_x = []
    train_data_y = []
    val_data_y = []
    for file in tqdm(os.listdir(DAILY_DIR)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        data = joblib.load(path)
        data = data[data["volume"] != 0]
        
        if data.index[-1] < to_date(last_trade_day): continue
        if len(data) <= 200: continue 

        # date = data.index
        # close = data.close.values
        # high = data.high.values
        # open = data.open.values
        # low = data.low.values
        # price = data.price.values
        # volume = data.volume.values
        # amount = data.amount.values
        # turn = data.turn.values
        
        x, y = (train_data_x, train_data_y) if np.random.random() < train_val_split else (val_data_x, val_data_y)
        
        
        train_data = data[train_features]
        
        date = train_data.index
        close = train_data.close.values
        high = train_data.high.values
        open = train_data.open.values
        low = train_data.low.values
        price = train_data.price.values
        volume = train_data.volume.values
        amount = train_data.amount.values
        turn = train_data.turn.values
        
        # print(train_data)
        up, nday, ratio = explain_label(train_label)
        train_labels = data[train_label].values
        
        for i in range(seq_len, len(train_data)-nday):
            # if amount[i] / turn[i] >= 50000000000: continue
            if high[i] / low[i] == 1: continue
            # if not three_days_increase_retr(i, open, close, high, low, price, turn): continue
            x.append(train_data.iloc[i-seq_len:i,:].values)
            y.append(train_labels[i])
            
            
    train_data_x = np.array(train_data_x)
    val_data_x = np.array(val_data_x)
    train_data_y = np.array(train_data_y)
    val_data_y = np.array(val_data_y)
    train_data_x = np.nan_to_num(train_data_x, 0)
    val_data_x = np.nan_to_num(val_data_x, 0)
    from collections import Counter
    print(Counter(train_data_y))
    np.savez("three_days_increase_train.npz", x=train_data_x, y=train_data_y)
    np.savez("three_days_increase_val.npz", x=val_data_x, y=val_data_y)
    print(train_data_x.shape)
    print(val_data_x.shape)

        
                
if __name__ == "__main__":
    generate()
    
    