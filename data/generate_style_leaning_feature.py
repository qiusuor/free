from utils import *
from config import *
from multiprocessing import Pool
from ydata_profiling import ProfileReport
import platform


def agg_groups(df):
    groups = {
        "limit_up": is_limit_up(df),
        "limit_down": is_limit_down(df),
        "limit_up_and_high_price_60": is_limit_up(df) & (df["price_div_chip_avg_60"] > 1.5),
        "limit_up_line": is_limit_up_line(df),
        "limit_down_line": is_limit_down_line(df),
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
        "mid_price_60": (df["price_div_chip_avg_60"] > 0.95) & (df["price_div_chip_avg_60"] <= 1.5),
        "low_price_60": df["price_div_chip_avg_60"] <= 0.95,
        "high_turn_60": df["turn_div_mean_turn_60"] > 2.5,
        "mid_turn_60": (df["turn_div_mean_turn_60"] > 0.75) & (df["turn_div_mean_turn_60"] <= 2.5),
        "low_turn_60": df["turn_div_mean_turn_60"] <= 0.75,
        "whole": df["price_div_chip_avg_60"] > -1,
        "up": df["close"] > df["preclose"],
        "down": df["close"] < df["preclose"],
        "red": df["close"] > df["open"],
        "blue": df["close"] < df["open"],
        "up_and_down": (df["high"] / df["close"] > 1.03) & (df["high"] / df["open"] > 1.05),
        "down_and_up": (df["low"] / df["close"] < 0.97) & (df["low"] / df["open"] < 0.95),
        "big_up": df["close"] / df["preclose"] > 1.05,
        "big_up_high_turn": (df["close"] / df["preclose"] > 1.05) & (df["turn_div_mean_turn_60"] > 2.5),
        "big_down": df["close"] / df["preclose"] < 0.94,
        "big_down_high_turn": (df["close"] / df["preclose"] < 0.94) & (df["turn_div_mean_turn_60"] > 2.5),
    }
    return groups

def stats_values(df, group_name, group):
    # observe = ["y_next_1d_ret", "y_next_1d_close_rate"]
    observe = ["y_next_1d_ret", "y_next_1d_close_rate", "y_next_1d_up_to_limit"]
    agg_methods = {
        "mean": np.mean, 
        "std": np.std, 
        # "max": np.max,
        # "min": np.min,
        # "num": len
    }
    
    group = df[group]
    group_agg_names = []
    group_agg_values = []
    for obs_name in observe:
        group_value = group[obs_name]
        for agg_name, agg_func in agg_methods.items():
            group_agg_names.append("_".join(["style_feat_shif1_of", obs_name, agg_name, group_name]))
            group_agg_values.append(agg_func(group_value) if len(group_value) else 0)
    return group_agg_names, group_agg_values
        

def generate_style_learning_info_one(argv):
    path, date = argv
    df = joblib.load(path)
    groups = agg_groups(df)
    agg_values = [date]
    agg_names = ["date"]
    for group_name, group in groups.items():
        group_agg_names, group_agg_values = stats_values(df, group_name, group)
        agg_values.extend(group_agg_values)
        agg_names.extend(group_agg_names)
    return agg_names, agg_values
    

def generate_style_learning_info():
    argvs = []
    for file in os.listdir(DAILY_BY_DATE_DIR):
        if not file.endswith(".pkl"): continue
        path = os.path.join(DAILY_BY_DATE_DIR, file)
        argvs.append((path, to_date(int(file[:-4]))))
    pool = Pool(THREAD_NUM)
    rets = pool.map(generate_style_learning_info_one, argvs)
    pool.close()
    pool.join()
    
    names = rets[0][0]
    values = list(list(zip(*rets))[1])
    df = pd.DataFrame(values, columns=names)
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    df = df.shift(1).fillna(0)
    # joblib.dump(df.columns, STYLE_FEAT_COLS)
    # keys = df.index.values
    # values = df.values
    # style_info = dict(zip(keys, values))
    # joblib.dump(style_info, STYLE_FEATS)
    
    # print(style_info)
    # print(list(df.columns))
    # exit(0)
    # df["label"] = df["y_next_1d_ret_mean_limit_up_and_high_price_60"].shift(-1) > 0.99
    joblib.dump(df, STYLE_FEATS)
    df.to_csv("style_learning_info.csv")
    merge_style_info()
    # print(df)
    
    # profile = ProfileReport(df[["y_next_1d_close_rate_mean_limit_up", "label"]], title="Style analyse")
    # profile.to_file("style_analyse.html")
    
def merge_style_info_one(path):
    style_feat = joblib.load(STYLE_FEATS)
    df = joblib.load(path)
    df = df.join(style_feat, how="left")
    df.to_csv(path.replace(".pkl", ".csv"))
    dump(df, path)
    
def merge_style_info():
    paths = []
    for file in tqdm(os.listdir(DAILY_DIR)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        paths.append(path)
    pool = Pool(8)
    pool.imap_unordered(merge_style_info_one, paths)
    pool.close()
    pool.join()
    
if __name__ == "__main__":
    generate_style_learning_info()
    