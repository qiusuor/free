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
        
        

if __name__ == "__main__":
    dir_a = "/home/qiusuo/free/data/data/daily"
    dir_b = "/home/qiusuo/free/data/data/daily_20231110"
    compare(dir_a, dir_b)