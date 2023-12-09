from utils import *
import joblib
from matplotlib import pyplot as plt
import lightgbm as lgb
import json
from generate_style_leaning_feature import generate_style_learning_info

def explore():
    generate_style_learning_info()
    df = joblib.load(STYLE_FEATS)
    df = df.fillna(0).astype(float)
    cols = [col for col in df.columns if "num" in col]
    render_html(df[cols], "style_learning_explore", "style_learning_explore.html")
    
if __name__ == "__main__":
    explore()
    
    