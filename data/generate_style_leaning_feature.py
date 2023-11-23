from utils import *
from config import *
from multiprocessing import Pool

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
        "whole": df["price_div_chip_avg_60"] > -1,
        "up": df["close"] > df["pre_close"],
        "down": df["close"] < df["pre_close"],
        "red": df["close"] > df["open"],
        "blue": df["close"] < df["open"],
        "up_and_down": (df["high"] / df["close"] > 1.03) & (df["high"] / df["open"] > 1.05),
        "down_and_up": (df["low"] / df["close"] < 0.97) & (df["low"] / df["open"] < 0.95),
        
    }
    return groups

def stats_values(df, group_name, group, date):
    observe = ["y_next_1d_ret", "y_next_1d_close_rate", "y_next_1d_up_to_limit"]
    agg_methods = {
        "mean": np.mean, 
        "std": np.std, 
        "max": np.max,
        "min": np.min,
    }
    
    group = df[group]
    group_agg_names = ["date"]
    group_agg_values = [date]
    for obs_name in observe:
        group_value = group[obs_name]
        for agg_name, agg_func in agg_methods.items():
            group_agg_names.append("_".join([obs_name, agg_name, group_name]))
            group_agg_values.append(agg_func(group_value) if len(group_value) else 0)
    return group_agg_names, group_agg_values
        

def generate_style_learning_info_one(argv):
    path, date = argv
    df = joblib.load(path)
    groups = agg_groups(df)
    agg_values = []
    agg_names = []
    for group_name, group in groups.items():
        group_agg_names, group_agg_values = stats_values(df, group_name, group, date)
        agg_values.extend(group_agg_values)
        agg_names.extend(group_agg_names)
    return agg_names, agg_values
    

def generate_style_learning_info():
    argvs = []
    for file in os.listdir(DAILY_BY_DATE_DIR):
        if not file.endswith(".pkl"): continue
        path = os.path.join(DAILY_BY_DATE_DIR, file)
        argvs.append((path, int(file[:-4])))
    generate_style_learning_info_one(argvs[0])
    pool = Pool(THREAD_NUM)
    rets = pool.map(generate_style_learning_info_one, argvs)
    pool.close()
    pool.join()
    
    names = rets[0][0]
    values = list(zip(*rets)[1])
    df = pd.DataFrame(values, columns=names)
    df.set_index("date")
    df.to_csv("style_learning_info.csv")
    
if __name__ == "__main__":
    generate_style_learning_info()
    