import joblib
from utils import *
from config import *
from tqdm import tqdm
import os

if __name__ == "__main__":
    
    for file in tqdm(os.listdir(DAILY_DIR)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        df = joblib.load(path)
        print(path)
        print(list(df.columns))
        break