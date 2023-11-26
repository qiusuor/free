import numpy as np
import pandas as pd
from talib import abstract
import talib
from multiprocessing import Pool
from config import *
from utils import *
from tqdm import tqdm
from joblib import dump
import warnings
from embedding.auto_encoder import MLPAutoEncoder
import platform

warnings.filterwarnings("ignore")


def inject_one(path):
    device = torch.device("mps") if platform.machine() == 'arm64' else torch.device("cuda")
    data = joblib.load(path)
    df = data[auto_encoder_features]
    torch.set_default_dtype(torch.float32)
    
    for batch_size, epochs, K, LAT_SIZE in auto_encoder_config:
        model = MLPAutoEncoder(input_size=len(auto_encoder_features)*K, lat_size=LAT_SIZE)
        model_path = "embedding/checkpoint/mlp_autoencoder_{}_{}.pth".format(K, LAT_SIZE)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        mean, std = joblib.load("embedding/checkpoint/mean_std_{}_{}.pkl".format(K, LAT_SIZE))
        mean = mean.to(device)
        std = std.to(device)
        with torch.no_grad():
            data_i = [df]
            for i in range(1, K):
                data_i.append(df.shift(i))
            data_i = data_i[::-1]
            data_i = pd.concat(data_i, axis=1)
            data_i = data_i.fillna(0).astype("float32").values
            data_i = torch.from_numpy(data_i).to(device)
            data_i = (data_i - mean) / (std + 1e-9)
            lat_i = model(data_i)[1]
            names = ["emb_{}_of_{}_{}".format(i, K, LAT_SIZE) for i in range(LAT_SIZE)]
            data[names] = lat_i.cpu().detach().numpy()
    data.to_csv(path.replace(".pkl", ".csv"))
    dump(data, path)
    
    
    
def inject_embedding():
    paths = []
    with torch.no_grad():
        for file in tqdm(os.listdir(DAILY_DIR)):
            code = file.split("_")[0]
            if not_concern(code) or is_index(code):
                continue
            if not file.endswith(".pkl"):
                continue
            path = os.path.join(DAILY_DIR, file)
            paths.append(path)
    # print(paths[0])
    # inject_one(paths[0])
    pool = Pool(16 if "Linux" in platform.platform() else 4)
    pool.imap_unordered(inject_one, paths)
    pool.close()
    pool.join()
            
     
if __name__ == "__main__":
    inject_embedding()
    