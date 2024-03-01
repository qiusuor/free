from config import *
from utils import *
import json

def seach(dir):
    for label in tqdm(os.listdir(dir)):
        if not label.startswith("y") and not label.startswith("dy"): continue
        label_dir = os.path.join(dir, label)
        for exp_config in os.listdir(label_dir):
            configured_exp_dir = os.path.join(label_dir, exp_config)
            if not os.path.isdir(configured_exp_dir): continue
            pred_days = list(filter(lambda x: os.path.isdir(os.path.join(configured_exp_dir, x)), os.listdir(configured_exp_dir)))
            for pred_day in pred_days:
                predt_day_dir = os.path.join(configured_exp_dir, pred_day)
                if not os.path.isdir(predt_day_dir): continue
                if not os.path.exists(os.path.join(predt_day_dir, "meta.json")):
                    finished = False
                    break
                model = joblib.load(os.path.join(predt_day_dir, "model.pkl"))
                feature_importance += model.feature_importance()
                feature_name = model.feature_name()
                meta = json.load(open(os.path.join(predt_day_dir, "meta.json")))
                epoch = meta["info"]["epoch"]