from config import *
import pandas as pd
from multiprocessing import Pool
import joblib
import bisect
import os
from utils import *
import _pickle as cPickle
    
K = [1, 2, 3, 5, 7, 9, 15, 22]
def generate():
    trade_days, last_trade_day = get_last_update_date()
    
    for k in K:
        train_data = []
        for file in os.listdir(DAILY_DIR):
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
            
            for i in range(k, len(close)):
                train_data.append([high[i-k:i], open[i-k:i], low[i-k:i], close[i-k:i], price[i-k:i], turn[i-k:i]])
            
        train_data = np.array(train_data)
        np.savez(KMER_RAR.format(k), train_data)
        print(train_data.shape)

        
                
if __name__ == "__main__":
    generate()
    
    