import numpy as np

def feature_transform(df):
    features_trans = {
        "open": np.log, 
        "high": np.log, "low", "close", "price", "turn", "volume", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"}
    