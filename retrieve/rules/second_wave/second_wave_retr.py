from config import *
from utils import *
import pandas as pd
import pickle

def get_label(price, i):
    for j in range(i+1, i+second_wave_wait_day+1):
        if j >= len(price): return -1
        if price[j] / price[i] > second_wave_target_upper:
            return 2
        elif price[j] / price[i] < second_wave_target_lower:
            return 0
    return 1

def second_wave_retr():
    trade_days, last_trade_day = get_last_update_date()
    retr_data = []
    labels = []
    describe = []
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
        if len(data) <= second_wave_min_trade_day: continue 
        
        date = data.index
        close = data.close.values
        price = data.price.values
        amount = data.amount.values
        turn = data.turn.values
        peTTM = data.peTTM.values
        end_index = bisect.bisect_left(date, to_date(last_trade_day))
        end_index = min(end_index, len(data)-1)
        
        for j in range(second_wave_feature_len-1, end_index):
            if price[j] >= max(price[j-second_wave_retr_main_wave_climp_day:j+second_wave_retr_delay_main_wave_day]) and price[j] / min(price[j-second_wave_retr_main_wave_climp_day:j]) >= second_wave_retr_main_wave_ratio:
                for i in range(j+second_wave_retr_delay_main_wave_day, j+second_wave_retr_within_main_wave_day):
                    if i >= len(price): break
                    if (sorted(price[i-second_wave_retr_continuous_climp_n_day+1:i+1]) == price[i-second_wave_retr_continuous_climp_n_day+1:i+1]).all():
                        if second_wave_retr_main_wave_drawback_lower <= price[i]/price[j] <= second_wave_retr_main_wave_drawback_upper:
                            index = slice(i-second_wave_feature_len+1, i+1)
                            feat = [
                                # close[index],
                                price[index],
                                turn[index],
                                amount[index],
                                peTTM[index]
                            ]
                            label = get_label(price, i)
                            retr_data.append([code, date[i], feat, label])
                            labels.append(label)
                            describe.append([code, date[i], label])
                            
    with open(second_wave_retr_file, 'wb') as out_f:
        pickle.dump(retr_data, out_f)
    
    describe = pd.DataFrame(describe, columns=["code", "date", "label"])
    describe.to_csv(second_wave_retr_des_file, index=False)
    from collections import Counter
    c = Counter(labels)
    print(c)


    

if __name__ == "__main__":
    second_wave_retr()
    