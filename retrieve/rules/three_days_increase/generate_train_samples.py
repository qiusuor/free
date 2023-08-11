from config import *
import pandas as pd
from multiprocessing import Pool
import joblib
import bisect
import os
from utils import *
import _pickle as cPickle
from tqdm import tqdm

seq_len = 120
hold_day = 2
expect_gain = 1.05
tolerent_pay = 0.97
train_val_split = 0.7
assert seq_len >= 3
assert hold_day >= 2

def three_days_increase_retr(i, open, close, high, low, price, turn):
    if not (close[i] > close[i-1] > close[i-2]): return False
    if not (turn[i] > turn[i-1] > turn[i-2]): return False
    if turn[i] / turn[i-2] >= 2: return False
    if turn[i] > 15: return False
    if close[i] / close[i-i] >1.08: return False
    return True
    
    
def get_label(i, open, close, high, low, price, turn):
    for j in range(i+2, i+hold_day+1):
        if high[j] / open[i+1] > expect_gain: return 2
        if low[j] / open[i+1] < tolerent_pay: return 0
    return 1

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

        date = data.index
        close = data.close.values
        high = data.high.values
        open = data.open.values
        low = data.low.values
        price = data.price.values
        volume = data.volume.values
        amount = data.amount.values
        turn = data.turn.values
        
        x, y = (train_data_x, train_data_y) if np.random.random() < train_val_split else (val_data_x, val_data_y)
        
        for i in range(seq_len, len(close)-hold_day):
            if amount[i] / turn[i] >= 50000000000: continue
            if high[i] / low[i] == 1: continue
            # if not three_days_increase_retr(i, open, close, high, low, price, turn): continue
            x.append([open[i-seq_len:i], close[i-seq_len:i], high[i-seq_len:i], low[i-seq_len:i], price[i-seq_len:i], turn[i-seq_len:i]])
            y.append(get_label(i, open, close, high, low, price, turn))
            
    train_data_x = np.array(train_data_x)
    val_data_x = np.array(val_data_x)
    train_data_y = np.array(train_data_y)
    val_data_y = np.array(val_data_y)
    from collections import Counter
    print(Counter(train_data_y))
    np.savez("three_days_increase_train.npz", x=train_data_x, y=train_data_y)
    np.savez("three_days_increase_val.npz", x=val_data_x, y=val_data_y)
    print(train_data_x.shape)
    print(val_data_x.shape)

        
                
if __name__ == "__main__":
    generate()
    
    