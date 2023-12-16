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
    "first_up_to_limit_time"
    "first_down_to_limit_time"
    "times_break_and_back_to_up_limit"
    "times_break_and_back_to_down_limit"
    "turn_on_limit_up_line"
    "turn_on_limit_down_line"
    "len_of_upper_shadow_line"
    "len_of_lower_shadow_line"
    "div_price"
    "div_chip"
    "max_turn",
    "min_turn",
    ""
    
    # feat = []
    # for day_price in day_prices:
    #     day_price = np.array([x/day_price[0] for x in day_price])
    #     day_price_diff = day_price[1:] - day_price[:-1]
    #     pos = sum(filter(lambda x:x>0, day_price_diff) or [0])
    #     neg = sum(filter(lambda x:x<0, day_price_diff) or [0])
    #     wave = pos * neg
    #     feat.append([wave, max(day_price), min(day_price)])
    # return feat
        
        

def prepare_one(path):
    "first_up_to_limit_time"
    "first_down_to_limit_time"
    "times_break_and_back_to_up_limit"
    "times_break_and_back_to_down_limit"
    "turn_on_limit_up_line"
    "turn_on_limit_down_line"
    "len_of_upper_shadow_line"
    "len_of_lower_shadow_line"
    "div_price"
    "div_chip"
    "max_turn",
    "min_turn",
    ""
    
    data = joblib.load(path)
    #TODO
    
    reference_daily_df = joblib.load()
    "date", "day", "code", "open", "high", "low", "close", "amount", "volume", "price", "preclose_day", "amount_day"
    inject_limit(data)
    
    # norlize
    for col in ["open", "high", "low", "close"]:
        data[col] = data[col] / data["preclose_day"]
    data["amount"] = data["amount"] / data["amount_day"]
    
    
    df = pd.DataFrame([x[1][["price", "amount"]].values.reshape(-1) for x in data.groupby("day")])
    date = [x[0] for x in data.groupby("day")]
    feat["date"] = date
    feat[["minutes_wave", "minutes_max", "minutes_min"]] = get_handcraft_feat([x[1]["price"].values.reshape(-1) for x in data.groupby("day")])
    feat_path = os.path.join(MINUTE_FEAT, os.path.basename(path))
    feat['date'] = pd.to_datetime(feat['date'])
    feat.set_index("date", inplace=True)
    if os.path.exists(feat_path):
        old_feat = joblib.load(feat_path)
        old_feat_index = set(old_feat.index)
        feat = pd.concat([old_feat, feat[[i not in old_feat_index for i in feat.index]]], axis=0)
    feat.sort_index()
    feat.to_csv(feat_path.replace(".pkl", ".csv"))
    dump(feat, feat_path)
    
    
def prepare():
    data_dir = TDX_MINUTE_DIR if not os.path.exists(os.path.join(MINUTE_FEAT, "sh.600000_5_2.csv")) else TDX_MINUTE_RECENT_DIR
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
    pool = Pool(8)
    pool.imap_unordered(prepare_one, paths)
    pool.close()
    pool.join()
        


if __name__ == "__main__":
    prepare()