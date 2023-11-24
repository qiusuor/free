from config import *
from utils import *
import shutil
import warnings

warnings.filterwarnings("ignore")

def dump_ltr_data(argv):
    date, df_i = argv
    path = os.path.join(DAILY_BY_DATE_DIR, "{}.pkl".format(date))
    df_i.to_csv(path.replace(".pkl", ".csv"))
    dump(df_i, path)
    
def train_val_data_filter(df):
    return df[reserve_n_last(not_limit_line(df).shift(-1)) & (df.isST != 1)]
    
def generate_ltr_data():
    remove_dir(DAILY_BY_DATE_DIR)
    make_dir(DAILY_BY_DATE_DIR)
    data = []
    for file in tqdm(os.listdir(DAILY_DIR)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        df = joblib.load(path)
        if "code_name" not in df.columns or not isinstance(df.code_name[-1], str) or "ST" in df.code_name[-1] or "st" in df.code_name[-1] or "sT" in df.code_name[-1]:
                continue
        if len(df) < 300: continue
        df = train_val_data_filter(df)
        data.append(df)
    df = pd.concat(data)
    
    data_by_date = []
    for i, df_i in tqdm(df.groupby("date")):
        
        df_i["y_ltr_2d_open_high_label"] = df_i["y_next_2_d_high_ratio"] 
        df_i["y_ltr_2d_open_high_label"][df_i["y_next_2_d_high_ratio"] < 1.02] = 0
        df_i["y_ltr_2d_open_high_label"][df_i["y_next_2_d_high_ratio"] >= 1.02] = 1
        df_i["y_ltr_2d_open_high_label"][df_i["y_next_2_d_high_ratio"] >= 1.05] = 2
        df_i["y_ltr_2d_open_high_label"][df_i["y_next_2_d_high_ratio"] >= 1.09] = 3
        df_i["y_ltr_2d_open_high_label"][df_i["y_next_2_d_ret"] >= 1.09] = 5
        df_i["y_ltr_2d_open_high_label"] = df_i["y_ltr_2d_open_high_label"].fillna(0)
        df_i["y_ltr_2d_open_high_label"] = df_i["y_ltr_2d_open_high_label"].astype(int)
        data_by_date.append([to_int_date(i), df_i])
        
    pool = Pool(THREAD_NUM)
    pool.imap_unordered(dump_ltr_data, data_by_date)
    pool.close()
    pool.join()
        
if __name__ == "__main__":
    generate_ltr_data()
    