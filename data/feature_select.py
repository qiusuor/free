from scipy.stats import entropy
from utils import *
from config import *
import bisect
import pandas as pd
from collections import Counter
import numpy as np

def get_quantile(series):
    return series.quantile(np.linspace(0.0, 1.0, 10)).values

def calc_bin_distribution_kl(series_all, series_train, series_val):
    bins = get_quantile(series_all)
    def map_to_bin(x):
        index = bisect.bisect_left(bins, x)
        return index
    def to_sorted_distribution(series):
        c = Counter(series)
        return [c[i] for i in range(len(bins))]
    
    series_train = to_sorted_distribution(series_train.apply(map_to_bin))
    series_val = to_sorted_distribution(series_val.apply(map_to_bin))
    kl = entropy(series_val, series_train)
    if np.isnan(kl):
        kl = np.inf
    return kl

def get_rel_dist(a, b):
    return abs(a - b) / max(min(a, b), 1e-9)
    
def analyse_single_feature(train_start_day, train_end_day, val_start_day, val_end_day, label):
    dataset = []
    groups = []
    dates = []
    for file in os.listdir(DAILY_BY_DATE_DIR):
        if not file.endswith(".pkl"): continue
        path = os.path.join(DAILY_BY_DATE_DIR, file)
        df = joblib.load(path)
        dataset.append(df)
        groups.append(len(df))
        dates.append(int(file[:-4]))
        
    # train_start_day = to_int_date(train_start_day)
    # train_end_day = to_int_date(train_end_day)
    # val_start_day = to_int_date(val_start_day)
    # val_end_day = to_int_date(val_end_day)
    
    train_dataset, val_dataset = [], []
    train_groups, val_groups = [], []
    for data_day, g, date in zip(dataset, groups, dates):
        if train_start_day <= date <= train_end_day:
            train_dataset.append(data_day)
            train_groups.append(g)
        elif val_start_day <= date <= val_end_day:
            val_dataset.append(data_day)
            val_groups.append(g)
    train_dataset = pd.concat(train_dataset)
    val_dataset = pd.concat(val_dataset)
    train_label = train_dataset[label]
    val_label = val_dataset[label]
    
    feature_alignment = []
    for feature in get_feature_cols():
        # all_data_feature = all_dateset[feature][all_data_label].astype(float)
        train_feature = train_dataset[feature][train_label].astype(float)
        val_feature = val_dataset[feature][val_label].astype(float)
        skew_dis = get_rel_dist(train_feature.skew(), val_feature.skew())
        kurt_dis = get_rel_dist(train_feature.kurt(), val_feature.kurt())
        mean_dis = get_rel_dist(train_feature.mean(), val_feature.mean())
        # kl =  calc_bin_distribution_kl(all_data_feature, train_feature, val_feature)
        feature_alignment.append([feature, mean_dis, skew_dis, kurt_dis])
        # print(feature, kl)
    feature_alignment = pd.DataFrame(feature_alignment, columns=["features", "mean_value_dist", "skew_dist", "kurt_dist"])
    feature_alignment = feature_alignment.sort_values(by="mean_value_dist", ascending=False)
    feature_alignment.to_csv("feature_kl.csv", index=False)
    print(list(feature_alignment[feature_alignment.mean_value_dist > 0.10].features))
        
if __name__ == "__main__":
    analyse_single_feature(20220824, 20230904, 20231025, 20231115, "y_02_109")
    