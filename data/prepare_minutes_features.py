import os
from config import *
from embedding.auto_encoder import MLPAutoEncoder
from utils import *
import platform

LAT_SIZE = 16

def inject_limit(df):
    df["limit_up_price"] = (df["preclose_day"] * 1.1).apply(floor2)
    df["limit_down_price"] = (df["preclose_day"] * 0.9).apply(ceil2)

def get_handcraft_feat(df):
    first_up_to_limit_time = 1e6
    first_down_to_limit_time = 1e6
    times_break_and_back_to_up_limit = 0
    times_break_and_back_to_down_limit = 0
    turn_on_limit_up_line = 0
    turn_on_limit_down_line = 0
    div_price_minu = 0
    div_chip_minu = 0
    max_turn_minu = df.turn.max()
    mean_turn_minu = df.turn.mean()
    
    day_price = df.price.iloc[0]
    limit_up_price = df.limit_up_price.iloc[0]
    limit_down_price = df.limit_down_price.iloc[0]
    open_up_limit_state = df.open.iloc[0] >= limit_up_price
    open_down_limit_state = df.open.iloc[0] <= limit_down_price
    cur_open_up_limit_state = open_up_limit_state
    cur_open_down_limit_state = open_down_limit_state
    minu_limit_up = df.minu_limit_up.values
    minu_limit_down = df.minu_limit_down.values
    turn = df.turn.values
    close = df.close.values
    N = len(minu_limit_up)
    
    for i in range(N):
        div_price_minu += (close[i] - day_price) ** 2
        div_chip_minu += (close[i] - day_price) ** 2 * turn[i] * 1000
        if minu_limit_up[i]:
            first_up_to_limit_time = min(i, first_up_to_limit_time)
            if not cur_open_up_limit_state:
                times_break_and_back_to_up_limit += 1
            cur_open_up_limit_state = True
            turn_on_limit_up_line += turn[i]
        else:
            cur_open_up_limit_state = False
            
        if minu_limit_down[i]:
            first_down_to_limit_time = min(i, first_down_to_limit_time)
            if not cur_open_down_limit_state:
                times_break_and_back_to_down_limit += 1
            cur_open_down_limit_state = True
            turn_on_limit_down_line += turn[i]
        else:
            cur_open_down_limit_state = False
    return max_turn_minu, mean_turn_minu, div_chip_minu, div_price_minu, turn_on_limit_down_line, turn_on_limit_up_line, times_break_and_back_to_down_limit, times_break_and_back_to_up_limit, first_down_to_limit_time, first_up_to_limit_time


def prepare_one(path):
    df = joblib.load(path)
    reference_daily_path = os.path.join(DAILY_DOWLOAD_DIR_NO_ADJUST, os.path.basename(path).replace("_1_3", "_d_3"))
    reference_daily_df = joblib.load(reference_daily_path)
    reference_daily_df["day"] = reference_daily_df.index
    reference_daily_df["preclose_day"] = reference_daily_df["preclose"]
    reference_daily_df["volume_day"] = reference_daily_df["volume"]
    reference_daily_df["turn_day"] = reference_daily_df["turn"]
    df = df.join(reference_daily_df[["day", "preclose_day", "volume_day", "turn_day", "price"]], how="left", on="day", rsuffix="r")
    "date", "day", "code", "open", "high", "low", "close", "volume", "preclose_day", "volume_day", "turn_day"
    inject_limit(df)
    df["minu_limit_up"] = df["close"] >= df["limit_up_price"]
    df["minu_limit_down"] = df["close"] <= df["limit_down_price"]
    for col in ["open", "high", "low", "close"]:
        df["{}_norm".format(col)] = df[col] / df["preclose_day"]
    df["turn"] = df["volume"] / df["volume_day"] * df["turn_day"]
    
    feat = []
    days = []
    for date_i, df_i in df.groupby("day"):
        days.append(date_i)
        feat.append(get_handcraft_feat(df_i))
        
        
    feat = pd.DataFrame(feat, columns=["max_turn_minu", "mean_turn_minu", "div_chip_minu", "div_price_minu", "turn_on_limit_down_line", "turn_on_limit_up_line", "times_break_and_back_to_down_limit", "times_break_and_back_to_up_limit", "first_down_to_limit_time", "first_up_to_limit_time"])
    feat["date"] = days
    feat_path = os.path.join(MINUTE_FEAT, os.path.basename(path))
    feat.to_csv(feat_path.replace(".pkl", ".csv"), index=False)
    dump(feat, feat_path)
    
    
def prepare():
    data_dir = MINUTE_DIR
    make_dir(MINUTE_FEAT)
    paths = []
    for file in tqdm(os.listdir(data_dir)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(data_dir, file)
        paths.append(path)
    # prepare_one(paths[0])
    # exit(0)
    pool = Pool(THREAD_NUM)
    pool.imap_unordered(prepare_one, paths)
    pool.close()
    pool.join()
        


if __name__ == "__main__":
    prepare()