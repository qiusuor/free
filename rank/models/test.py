import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_curve, auc, average_precision_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from joblib import load, dump
import _pickle as cPickle
from multiprocessing import Process
from utils import *
from data.fetch_daily import fetch_daily
from data.inject_features import inject_features
from data.inject_labels import inject_labels
from matplotlib import pyplot as plt
import shutil
import json

# for path in main_board_stocks():
#     path = os.path.join(DAILY_DIR, file)
#     df = joblib.load(path)
#     df["date"] = df.index
#     print(list(df.columns))
#     print(df.date)
#     exit(0)

model = joblib.load("/home/qiusuo/free/rank/exp_pred/y_2_d_high_rank_20%_safe_1d/180_31_9_21/20230919/model.pkl")
print(model.feature_importance())
print(model.feature_name())