import os
from config import *
from embedding.auto_encoder import MLPAutoEncoder
from utils import *
import platform

LAT_SIZE = 16

def get_handcraft_feat(day_prices):
    feat = []
    for day_price in day_prices:
        day_price = np.array([x/day_price[0] for x in day_price])
        day_price_diff = day_price[1:] - day_price[:-1]
        pos = sum(filter(lambda x:x>0, day_price_diff) or [0])
        neg = sum(filter(lambda x:x<0, day_price_diff) or [0])
        wave = pos * neg
        feat.append([wave, max(day_price), min(day_price)])
    return feat
        
        

def prepare_one(path):
    device = torch.device("mps") if platform.machine() == 'arm64' else torch.device("cuda")
    data = joblib.load(path)
    torch.set_default_dtype(torch.float32)
    df = pd.DataFrame([x[1][["price", "amount"]].values.reshape(-1) for x in data.groupby("day")])
    date = [x[0] for x in data.groupby("day")]
    model = MLPAutoEncoder(input_size=96, lat_size=LAT_SIZE, hidden_size=32)
    model_path = "embedding/checkpoint/minutes_mlp_autoencoder_{}.pth".format(LAT_SIZE)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    mean, std = joblib.load("embedding/checkpoint/minutes_mean_std_{}.pkl".format(LAT_SIZE))
    mean = mean.to(device)
    std = std.to(device)
    df = torch.tensor(df.values).cuda().float()
    df = (df -mean) / (std + 1e-9)
    names = ["minutes_emb_{}_of_{}".format(i, LAT_SIZE) for i in range(LAT_SIZE)]
    with torch.no_grad():
        lat_i = model(df)[1]
        feat = pd.DataFrame(lat_i.cpu().numpy(), columns=names)
    feat["date"] = date
    feat[["minutes_wave", "minutes_max", "minutes_min"]] = get_handcraft_feat([x[1]["price"].values.reshape(-1) for x in data.groupby("day")])
    feat_path = os.path.join(MINUTE_FEAT, os.path.basename(path))
    feat['date'] = pd.to_datetime(feat['date'])
    feat.set_index("date", inplace=True)
    if os.path.exists(feat_path):
        old_feat = joblib.load(feat_path)
        old_feat_index = set(old_feat.index)
        feat = pd.concat([old_feat, feat[[i not in old_feat_index for i in feat.index]]], axis=0)
    feat.to_csv(feat_path.replace(".pkl", ".csv"))
    dump(feat, feat_path)
    
    
def prepare():
    data_dir = MINUTE_DIR if not os.path.exists(os.path.join(MINUTE_FEAT, "sh.600000_5_2.csv")) else MINUTE_RECENT_DIR
    make_dir(data_dir)
    paths = []
    for file in tqdm(os.listdir(data_dir)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(data_dir, file)
        paths.append(path)
    # print(paths[0])
    # prepare_one(paths[0])
    # exit(0)
    pool = Pool(32)
    pool.imap_unordered(prepare_one, paths)
    pool.close()
    pool.join()
        


if __name__ == "__main__":
    prepare()