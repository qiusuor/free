from utils import *
from config import *
from multiprocessing import Pool


def agg_groups(df):
    groups = {
        "limit_up": (df["limit_up"], 1),
        "limit_up_1d": (df["limit_up_1d"] & (~df["limit_up_line"]), ~df["limit_up_pre_day"].astype(bool)),
        "limit_up_2d": (df["limit_up_2d"] & (~df["limit_up_line"]), df["limit_up_1d_pre_day"]),
        "limit_up_3d": (df["limit_up_3d"] & (~df["limit_up_line"]), df["limit_up_2d_pre_day"]),
        "limit_up_4d": (df["limit_up_4d"] & (~df["limit_up_line"]), df["limit_up_3d_pre_day"]),
        "limit_up_5d": (df["limit_up_5d"] & (~df["limit_up_line"]), df["limit_up_4d_pre_day"]),
        "limit_up_6d": (df["limit_up_6d"] & (~df["limit_up_line"]), df["limit_up_5d_pre_day"]),
        "limit_up_7d": (df["limit_up_7d"] & (~df["limit_up_line"]), df["limit_up_6d_pre_day"]),
        "limit_up_8d": (df["limit_up_8d"] & (~df["limit_up_line"]), df["limit_up_7d_pre_day"]),
        "limit_up_9d": (df["limit_up_9d"] & (~df["limit_up_line"]), df["limit_up_8d_pre_day"]),
        "limit_up_high": (df["limit_up_high"] & (~df["limit_up_line"]), df["limit_up_pre_day"].astype(bool) & ~df["limit_up_1d_pre_day"]),
        # "limit_up_9d_plus": (df["limit_up_9d_plus"], 1),
        # "limit_up_line": (df["limit_up_line"], 1),
        "limit_up_line_1d": (df["limit_up_line_1d"], ~df["limit_up_pre_day"].astype(bool)),
        "limit_up_line_2d": (df["limit_up_line_2d"], df["limit_up_line_1d_pre_day"]),
        "limit_up_line_3d": (df["limit_up_line_3d"], df["limit_up_line_2d_pre_day"]),
        "limit_up_line_4d": (df["limit_up_line_4d"], df["limit_up_line_3d_pre_day"]),
        "limit_up_line_5d": (df["limit_up_line_5d"], df["limit_up_line_4d_pre_day"]),
        "limit_up_line_6d": (df["limit_up_line_6d"], df["limit_up_line_5d_pre_day"]),
        "limit_up_line_7d": (df["limit_up_line_7d"], df["limit_up_line_6d_pre_day"]),
        
        # "limit_down": (df["limit_down"], 1),
        "limit_down_1d": (df["limit_down_1d"] & (~df["limit_down_line"]), ~df["limit_down_pre_day"].astype(bool)),
        "limit_down_2d": (df["limit_down_2d"] & (~df["limit_down_line"]), df["limit_down_1d_pre_day"]),
        "limit_down_3d": (df["limit_down_3d"] & (~df["limit_down_line"]), df["limit_down_2d_pre_day"]),
        "limit_down_4d": (df["limit_down_4d"] & (~df["limit_down_line"]), df["limit_down_3d_pre_day"]),
        "limit_down_5d": (df["limit_down_5d"] & (~df["limit_down_line"]), df["limit_down_4d_pre_day"]),
        # "limit_down_5d_plus": df["limit_down_5d_plus"],
        # "limit_down_line": (df["limit_down_line"], 1),
        "limit_down_line_1d": (df["limit_down_line_1d"], ~df["limit_down_line_pre_day"].astype(bool)),
        "limit_down_line_2d": (df["limit_down_line_2d"], df["limit_down_line_1d_pre_day"]),
        "limit_down_line_3d": (df["limit_down_line_3d"], df["limit_down_line_2d_pre_day"]),
        "limit_down_line_4d": (df["limit_down_line_4d"], df["limit_down_line_3d_pre_day"]),
        
        # "high_price_60": df["price_div_chip_avg_60"] > 1.25,
        # "high_turn_60": df["turn_div_mean_turn_60"] > 3.5,
    }
    return groups

def stats_values(df, group_name, group, date, mask):
    observe = ["y_next_1d_ret", "y_open_close", "y_next_1d_ret_close"]
    agg_methods = {
        "mean": np.mean, 
        "std": np.std, 
        "num": len
    }
    
    default_value = {
        "mean": 1,
        "std": 0,
        "num": 0
    }
    group = df[group]
    group_agg_names = []
    group_agg_values = []
    for obs_name in observe:
        group_value = group[obs_name]
        limit_value = df[group_name]
        limit_value = limit_value[df.reach_limit_up & mask]
        group_agg_names.append("_".join(["style_feat", obs_name, "close_rate", group_name]))
        agg_value = (limit_value.astype(float).mean() - 1)
        # if np.isnan(agg_value):
        #     agg_value = -0.5
        group_agg_values.append(agg_value)
        # if np.isnan(agg_value) and group_name == "limit_up_1d":
        #     print(group_name, agg_value, limit_value, date)
        for agg_name, agg_func in agg_methods.items():
            group_agg_names.append("_".join(["style_feat", obs_name, agg_name, group_name]))
            agg_value = agg_func(group_value) if len(group_value) else default_value[agg_name]
            if agg_name == "mean":
                agg_value -= 1
            group_agg_values.append(agg_value)

    return group_agg_names, group_agg_values
        

def generate_style_learning_info_one(argv):
    path, date = argv
    df = joblib.load(path)
    groups = agg_groups(df)
    agg_values = [date]
    agg_names = ["date"]
    for group_name, (group, mask) in groups.items():
        group_agg_names, group_agg_values = stats_values(df, group_name, group, date, mask)
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
    df = df.shift(1)
    df = df.iloc[2:]
    joblib.dump(df, STYLE_FEATS)
    df.to_csv(STYLE_FEATS.replace(".pkl", ".csv"))
    merge_style_info()
    
def merge_style_info_one(path):
    style_feat = joblib.load(STYLE_FEATS)
    df = joblib.load(path)
    df = df.join(style_feat, how="left")
    df.to_csv(path.replace(".pkl", ".csv"))
    dump(df, path)
    
def merge_style_info():
    paths = main_board_stocks()
    pool = Pool(THREAD_NUM)
    pool.imap_unordered(merge_style_info_one, paths)
    pool.close()
    pool.join()
    
if __name__ == "__main__":
    generate_style_learning_info()
    