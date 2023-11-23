from utils import *
from config import *

def agg_groups(df):
    groups = {
        "limit_up_1d": df["limit_up_1d"],
        "limit_up_2d": df["limit_up_2d"],
        "limit_up_3d": df["limit_up_3d"],
        "limit_up_4d": df["limit_up_4d"],
        "limit_up_5d": df["limit_up_5d"],
        "limit_up_5_plus_d": df["limit_up_5_plus_d"],
        "limit_down_1d": df["limit_down_1d"],
        "limit_down_2d": df["limit_down_2d"],
        "limit_down_3d": df["limit_down_3d"],
        "high_price_60": df["price_div_chip_avg_60"] > 1.5,
        "mid_price_60": (df["price_div_chip_avg_60"] > 1.0) & (df["price_div_chip_avg_60"] <= 1.5),
        "low_price_60": df["price_div_chip_avg_60"] <= 1.0,
        "high_turn_60": df["turn_div_mean_turn_60"] > 2.5,
        "mid_turn_60": (df["turn_div_mean_turn_60"] > 1) & (df["turn_div_mean_turn_60"] <= 2.5),
        "low_turn_60": df["turn_div_mean_turn_60"] <= 1,
        
        
    }
    return groups

def stats_values(df):
    pass

def generate_one(df):
    pass


def generate_style_learning_feature():
    dataset = []
    groups = []
    dates = []
    for file in os.listdir(DAILY_BY_DATE_DIR):
        if not file.endswith(".pkl"): continue
        path = os.path.join(DAILY_BY_DATE_DIR, file)
        df = joblib.load(path)
        dataset.append(df)
        groups.append(len(df))
        dates.append(int(file[:-4]))
   