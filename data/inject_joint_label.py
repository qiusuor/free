import numpy as np
import pandas as pd
from talib import abstract
import talib
from multiprocessing import Pool
from config import *
from utils import *
from tqdm import tqdm
from joblib import dump
import warnings

warnings.filterwarnings("ignore")

def dump_i(argv):
    i, df_i = argv
    df_i = df_i.sort_index()
    path = os.path.join(DAILY_DIR, "{}_d_2.pkl".format(i))
    df_i.to_csv(path.replace(".pkl", ".csv"))
    dump(df_i, path)
    
def injecto_joint_label():
    data = []
    for file in tqdm(os.listdir(DAILY_DIR)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        df = joblib.load(path)
        data.append(df)
    df = pd.concat(data)
    
    data = []
    for i, df_i in df.groupby("date"):
        for d in [2, 3, 5, 10, 22]:
            df_i["y_{}_d_high_rank".format(d)] = df_i["y_next_{}_d_high_ratio".format(d)].rank(pct=True, ascending=False)
            df_i["y_{}_d_high_rank_10%".format(d)] = (df_i["y_{}_d_high_rank".format(d)] <= 0.1).astype("float")
            df_i["y_{}_d_high_rank_20%".format(d)] = (df_i["y_{}_d_high_rank".format(d)] <= 0.2).astype("float")
            df_i["y_{}_d_high_rank_30%".format(d)] = (df_i["y_{}_d_high_rank".format(d)] <= 0.3).astype("float")
            df_i["y_{}_d_high_rank_50%".format(d)] = (df_i["y_{}_d_high_rank".format(d)] <= 0.5).astype("float")
            df_i["y_{}_d_ret_rank".format(d)] = df_i["y_next_{}_d_ret".format(d)].rank(pct=True, ascending=False)
            df_i["y_{}_d_ret_rank_10%".format(d)] = (df_i["y_{}_d_ret_rank".format(d)] <= 0.1).astype("float")
            df_i["y_{}_d_ret_rank_20%".format(d)] = (df_i["y_{}_d_ret_rank".format(d)] <= 0.2).astype("float")
            df_i["y_{}_d_ret_rank_30%".format(d)] = (df_i["y_{}_d_ret_rank".format(d)] <= 0.3).astype("float")
            df_i["y_{}_d_ret_rank_50%".format(d)] = (df_i["y_{}_d_ret_rank".format(d)] <= 0.5).astype("float")
        data.append(df_i)
        
    df = pd.concat(data)
    
    pool = Pool(THREAD_NUM)
    pool.imap_unordered(dump_i, df.groupby("code"))
    pool.close()
    pool.join()


if __name__ == "__main__":
    injecto_joint_label()
    