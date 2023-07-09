from config import *
import pandas as pd
from multiprocessing import Pool
import joblib
import bisect
import os
from utils import *
from _pickle import cPickle
    
  
def generate():
    trade_days, last_trade_day = get_last_update_date()
    for file in os.listdir(DAILY_DIR):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        data = joblib.load(path)
        data = data[data["volume"] != 0]
        
        if data.index[-1] < last_trade_day: continue
        if len(data) <= 200: continue 

        date = data.index
        close = data.close.values
        high = data.high.values
        open = data.open.values
        low = data.low.values
        price = data.price.values
        volume = data.volume.values
        amount = data.amount.values
        
        
                
if __name__ == "__main__":
    generate()
    
    