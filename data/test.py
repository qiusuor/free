from utils import *

# inject_joint_label()

# get_industry_info()

# get_shibor()
# get_profit("sh.600533", False)

# import pandas as pd

# a = pd.DataFrame([[2, 2], [3, 3]], columns=["a", "b"])
# b = pd.DataFrame([[1, 11], [2, 22], [3, 33]], columns=["a", "c"])
# a.set_index("a", inplace=True)
# b.set_index("a", inplace=True)

# print(a)
# print(b)
# a["c"]=b["c"]
# print(a)

def check_same(dfa, dfb):
    cols = dfa.columns
    N = len(dfa)
    index = dfa.index
    dfb = dfb[cols]
    for i in range(N):
        val = dfa.iloc[i]
        vbl = dfb.iloc[i]
        for va, vb, col in zip(val, vbl, cols):
            # print(va, vb)
            if col.startswith("y") or col.startswith("dy"):
                if i >= N-2: continue
            if va != vb and not(np.isnan(va) and np.isnan(vb)):
                # if if np.isnan(va) and np.isnan(vb):
                print(va, vb, col, index[i], len(cols), len(val), len(vbl))
                assert False
        
        
def compare(dir_a, dir_b):
    for file in tqdm(os.listdir(dir_a)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        patha = os.path.join(dir_a, file)
        pathb = os.path.join(dir_b, file)
        if not os.path.exists(pathb): continue
        dfa = joblib.load(patha)
        dfb = joblib.load(pathb)
        print(patha, pathb)
        index = dfa.index
        dfb = dfb.loc[index]
        check_same(dfa, dfb)
        # print(dfa)
        # print(dfb)
        # exit(0)
        
        
def check_feature_importance():
    model_a = joblib.load("/home/qiusuo/free/rank/exp_rank_pred/y_ltr_2d_open_high_label/250_31_9_81/20231018/model.pkl")
    model_b = joblib.load("/home/qiusuo/free/rank/exp_rank_pred/y_ltr_2d_open_high_label/250_31_9_81/20231023/model.pkl")
    
    def print_model_feature_importance(model):
        print(list(zip(model.feature_importance(), model.feature_name())))
    print_model_feature_importance(model_a)
    print_model_feature_importance(model_b)

if __name__ == "__main__":
    # dir_a = "/home/qiusuo/free/data/data/daily"
    # dir_b = "/home/qiusuo/free/data/data/daily_20231110"
    # compare(dir_a, dir_b)
    # check_feature_importance()
    upload_data_and_run()