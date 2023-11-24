from utils import *
import joblib
from matplotlib import pyplot as plt
import lightgbm as lgb
import json

def train():
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {"auc"},
        'num_leaves': 31,
        "min_data_in_leaf": 3,
        'learning_rate': 0.05,
        'feature_fraction': 0.99,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'verbose': 1,
        "train_metric": True,
        "max_depth": 9,
        "num_iterations": 100,
        "early_stopping_rounds": 20,
        "min_gain_to_split": 0,
        "num_threads": 8,
    }
    
    df = joblib.load("style_learning_info.pkl")
    df = df.fillna(0).astype(float)
    train_data, val_data, test_data = df.iloc[:-60], df.iloc[-60:-30], df.iloc[-30:]
    train_y = train_data["label"]
    train_x = train_data.drop("label", axis=1)
    val_y = val_data["label"]
    val_x = val_data.drop("label", axis=1)
    test_y = test_data["label"]
    test_x = test_data.drop("label", axis=1)
    
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_val = lgb.Dataset(val_x, val_y, reference=lgb_train)
    lgb_test = lgb.Dataset(test_x, test_y, reference=lgb_train)
    gbm = lgb.train(params,
                lgb_train,
                valid_sets=(lgb_train, lgb_val, lgb_test),
                )
    gbm.save_model("model.txt")
    lgb.plot_importance(gbm, max_num_features=50, height=0.5, figsize=(30, 18))
    plt.title("Feature importances top-50")
    plt.savefig("FeatureImportancesForStyleLearning.png")
  
if __name__ == "__main__":
    train()
    