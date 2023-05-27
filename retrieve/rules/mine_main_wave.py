from config import *
import pandas as pd
from multiprocessing import Pool
import joblib
import bisect
import os
from utils import *
    
def mine(start_day, end_day, recent_day, chip_slice_len, exp_decay_rate, increase_day=30, increase_rate=2.0):
    main_wave = []
    columns = ["code", "start_day", "increase_rate"]
    for file in os.listdir(DAILY_DIR):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        data = joblib.load(path)
        data = data[data["volume"] != 0]
        
        if data.index[-1] < end_day: continue
        if len(data) <= 200: continue 

        date = data.index
        close = data.close.values
        high = data.high.values
        open = data.open.values
        low = data.low.values
        price = data.price.values
        volume = data.volume.values
        amount = data.amount.values
        start_index = bisect.bisect_left(date, start_day)
        start_index = max(start_index, increase_day)
        end_index = bisect.bisect_left(date, end_day)
        
        for i in range(start_index, end_index-1):
            if max(price[i+1:i+increase_day+1]) / price[i] > increase_rate:
                main_wave.append([code, date[i], max(price[i+1:i+increase_day+1]) / price[i]])

    main_wave = pd.DataFrame(data=main_wave, columns=columns)
    main_wave.to_csv(GT_MAIN_WAVE, index=False)
    
def mine_main_wave():
    trade_days, last_trade_day = get_last_update_date()
    start_day, end_day, recent_day, chip_slice_len, exp_decay_rate = to_date(20220101), to_date(last_trade_day), 250, 120, 0.997
    mine(start_day, end_day, recent_day, chip_slice_len, exp_decay_rate)
                
if __name__ == "__main__":
    mine_main_wave()
    
    