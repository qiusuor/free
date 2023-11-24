from utils import *
import joblib
from matplotlib import pyplot as plt
import lightgbm as lgb
import json
from generate_style_leaning_feature import generate_style_learning_info

def explore():
    generate_style_learning_info()
    df = joblib.load("style_learning_info.pkl")
    df = df.fillna(0).astype(float)
    render_html(df, "style_learning_explore", "style_learning_explore.html")
    
if __name__ == "__main__":
    explore()
    
    