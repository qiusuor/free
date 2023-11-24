from utils import *
import joblib
from matplotlib import pyplot as plt
import lightgbm as lgb
import json

def explore():
    
    df = joblib.load("style_learning_info.pkl")
    df = df.fillna(0).astype(float)
    plot_cols = ["y_next_1d_ret_mean_limit_up", "y_next_1d_ret_std_limit_up"]
    render_html(df[plot_cols], "style_learning_explore", "style_learning_explore.html")
    
if __name__ == "__main__":
    explore()
    
    