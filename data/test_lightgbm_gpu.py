import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import time
 
def run_lightgbm():
    
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
 
    clf = lgb.LGBMClassifier(learning_rate=0.1,
                            n_estimators=10000,
                            device='gpu',
                            gpu_platform_id=0,
                            gpu_device_id=0,
                            verbose=1)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])
 
 
def run_xgboost():
    import xgboost as xgb
 
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        missing=-999,
        random_state=2019,
        gpu_id=0,
        tree_method='gpu_hist'
    )
    clf.fit(X_train, y_train, eval_metric=['error'])
 
 
def run_catboost():
    import catboost
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = catboost.CatBoostClassifier(verbose=100, iterations=300, learning_rate=0.001, task_type='GPU')
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])
 
 
if __name__ == "__main__":
    t_start = time.time()
    run_lightgbm()
    print(time.time() - t_start)
    