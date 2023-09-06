from config import *
from utils import *
import pandas as pd
import pickle
from data.plot_utils import plot_kline_volume

MIN_HIS = 7

def up_and_down_retr():
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
        data = data[data["price"] != 0]
        
        date = data.index
        close = data.close.values
        price = data.price.values
        amount = data.amount.values
        turn = data.turn.values
        peTTM = data.peTTM.values
        
        N = len(date)
        
        for i in range(N)
            
                            


    

if __name__ == "__main__":
    up_and_down_retr()
    