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
    
    
def mv():
    for src in os.listdir(MINUTE_DIR):
        trt = src.replace("_5_2", "_5_3")
        os.system("mv {} {}".format(os.path.join(MINUTE_DIR, src), os.path.join(MINUTE_DIR, trt)))
  
dirs = [r"C:\Users\qiusuo\Desktop\2021", r"C:\Users\qiusuo\Desktop\2022", r"C:\Users\qiusuo\Desktop\2023"]

def process_one(code):
    print(code)
    code_df = []
    if not code.endswith(".csv"): return
    for dir in dirs:
        if os.path.exists(os.path.join(dir, code)):
            code_df.append(pd.read_csv(os.path.join(dir, code)))
    code_df = pd.concat(code_df)
    code_df["date"] = pd.to_datetime(code_df["trade_time"])
    code_df["volume"] = code_df["vol"]
    code_df["day"] = code_df["date"].apply(lambda x: to_date(x.strftime("%Y-%m-%d")))
    
    code_df = code_df.sort_values(by="date")
    code_df = code_df[["date", "day", "open", "high", "low", "close", "volume"]]
    
    code_file_name = "sz.{}_1_3.csv".format(code[:6]) if "SZ" in code else "sh.{}_1_3.csv".format(code[:6])
    path = os.path.join(MINUTE_DIR, code_file_name)
    code_df.to_csv(path, index=False)
    joblib.dump(code_df, path.replace("csv", "pkl"))
              

def merge_minutes():
    make_dir(MINUTE_DIR)
    
    "trade_time,open,high,low,close,vol,amount"
    all_codes = []
    for dir in dirs:
        all_codes.extend(os.listdir(dir))
    all_codes = list(set(all_codes))
        
    # for code in tqdm(all_codes):

        
    process_one(all_codes[0])
    pool = Pool(THREAD_NUM)
    pool.imap_unordered(process_one, all_codes)
    pool.close()
    pool.join()
        

if __name__ == "__main__":
    # dir_a = "/home/qiusuo/free/data/data/daily"
    # dir_b = "/home/qiusuo/free/data/data/daily_20231110"
    # compare(dir_a, dir_b)
    # check_feature_importance()
    # upload_data()
    # mv()
    merge_minutes()