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
from embedding.auto_encoder import MLPAutoEncoder
import platform

warnings.filterwarnings("ignore")


def inject_one(path):
    code = os.path.basename(path).split("_")[0]
    data = joblib.load(path)
    minutes_feat_path = os.path.join(MINUTE_FEAT, "{}_5_2.pkl".format(code))
    minutes_feat = joblib.load(minutes_feat_path)
    data = data.join(minutes_feat)
    data.to_csv(path.replace(".pkl", ".csv"))
    dump(data, path)
    
    
    
def inject_minute_feature():
    paths = main_board_stocks()
    # print(paths[0])
    # inject_one(paths[0])
    pool = Pool(32)
    pool.imap_unordered(inject_one, paths)
    pool.close()
    pool.join()
            
     
if __name__ == "__main__":
    inject_minute_feature()
    
    