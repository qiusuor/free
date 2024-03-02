from config import *
from utils import *
import json

def seach(dir=EXP_RANK_DIR):
    agg_result = dict()
    agg_result["all_exp"] = []
    for label in tqdm(os.listdir(dir)):
        if not label.startswith("y") and not label.startswith("dy"): continue
        label_dir = os.path.join(dir, label)
        for exp_config in os.listdir(label_dir):
            configured_exp_dir = os.path.join(label_dir, exp_config)
            if not os.path.isdir(configured_exp_dir): continue
            pred_days = list(filter(lambda x: os.path.isdir(os.path.join(configured_exp_dir, x)), os.listdir(configured_exp_dir)))
            feature_importance = 0
            
            epochs = []
            val_ndcg_3, val_ndcg_5, val_ndcg_10, val_ndcg_30, val_ndcg_50, val_ndcg_100, val_ndcg_200 = [], [], [], [], [], [], []
            for pred_day in pred_days:
                predt_day_dir = os.path.join(configured_exp_dir, pred_day)
                if not os.path.isdir(predt_day_dir): continue
                model = joblib.load(os.path.join(predt_day_dir, "model.pkl"))
                feature_name = model.feature_name()
                meta = json.load(open(os.path.join(predt_day_dir, "meta.json")))
                epoch = meta["info"]["epoch"]
                epochs.append(epoch)
                val_ndcg_3.append(meta["info"]["val_ndcg_3"])
                val_ndcg_5.append(meta["info"]["val_ndcg_5"])
                val_ndcg_10.append(meta["info"]["val_ndcg_10"])
                val_ndcg_30.append(meta["info"]["val_ndcg_30"])
                val_ndcg_50.append(meta["info"]["val_ndcg_50"])
                val_ndcg_100.append(meta["info"]["val_ndcg_100"])
                val_ndcg_200.append(meta["info"]["val_ndcg_200"])
                feature_importance += model.feature_importance()
                feature_name = model.feature_name()
                
            feature_importance, feature_name = zip(*sorted(list(zip(feature_importance, feature_name)), key=lambda x:-x[0]))
            
            exp_result = {
                "label": label,
                "exp_config": exp_config,
                "avg_epoch": np.mean(epochs),
                "val_ndcg_3": np.mean(val_ndcg_3),
                "val_ndcg_5": np.mean(val_ndcg_5),
                "val_ndcg_10": np.mean(val_ndcg_10),
                "val_ndcg_30": np.mean(val_ndcg_30),
                "val_ndcg_50": np.mean(val_ndcg_50),
                "val_ndcg_100": np.mean(val_ndcg_100),
                "val_ndcg_200": np.mean(val_ndcg_200),
                "feature_importance": str(feature_importance[:50]),
                "feature_name": str(feature_name[:50])
            }
            agg_result["all_exp"].append(exp_result)
    agg_result["val_ndcg_3"] = sorted(agg_result["all_exp"], key=lambda x:-x["val_ndcg_3"])
    agg_result["val_ndcg_5"] = sorted(agg_result["all_exp"], key=lambda x:-x["val_ndcg_5"])
    agg_result["val_ndcg_10"] = sorted(agg_result["all_exp"], key=lambda x:-x["val_ndcg_10"])
    agg_result["val_ndcg_30"] = sorted(agg_result["all_exp"], key=lambda x:-x["val_ndcg_30"])
    agg_result["val_ndcg_50"] = sorted(agg_result["all_exp"], key=lambda x:-x["val_ndcg_50"])
    agg_result["val_ndcg_100"] = sorted(agg_result["all_exp"], key=lambda x:-x["val_ndcg_100"])
    agg_result["val_ndcg_200"] = sorted(agg_result["all_exp"], key=lambda x:-x["val_ndcg_200"])
    with open(os.path.join(dir, "result.json"), 'w') as f:
        json.dump(agg_result, f, indent=2)
               
            
if __name__ == "__main__":
    seach()