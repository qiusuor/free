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
from data.fetch import fetch_daily
from data.inject_features import inject_features
from data.inject_labels import inject_labels
from matplotlib import pyplot as plt
import shutil
import json

for file in os.listdir(DAILY_DIR):
    code = file.split("_")[0]
    if not_concern(code) or is_index(code):
        continue
    if not file.endswith(".pkl"):
        continue
    path = os.path.join(DAILY_DIR, file)
    df = joblib.load(path)
    df["date"] = df.index
    print(list(df.columns))
    print(df.date)
    exit(0)