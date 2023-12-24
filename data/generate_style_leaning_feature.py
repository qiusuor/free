from utils import *
from config import *
from multiprocessing import Pool


def agg_groups(df):
    groups = {
        "limit_up": df["limit_up_1d"],
        "limit_down": df["limit_down_1d"],
        "limit_up_line": df["limit_up_line"],
        "limit_down_line": df["limit_down_line"],
        "high_price_60": df["price_div_chip_avg_60"] > 1.25,
        "high_turn_60": df["turn_div_mean_turn_60"] > 3.5,
    }
    return groups

def stats_values(df, group_name, group):
    observe = ["y_next_1d_ret"]
    agg_methods = {
        "mean": np.mean, 
        "std": np.std, 
        "num": len
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
    paths = []
    for file in tqdm(os.listdir(DAILY_DIR)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        paths.append(path)
    pool = Pool(THREAD_NUM)
    pool.imap_unordered(merge_style_info_one, paths)
    pool.close()
    pool.join()
    
if __name__ == "__main__":
    generate_style_learning_info()
    