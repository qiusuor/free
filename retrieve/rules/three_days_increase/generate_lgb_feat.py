from config import *
import pandas as pd
import multiprocessing
from multiprocessing import Process
import joblib
import bisect
import os
from utils import *
import _pickle as cPickle
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

seq_len = 5
expect_gain = 1.08
tolerent_pay = 0.97
train_val_split = 0.7
assert seq_len >= 3

train_features =['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM', 'isST', 'factor', 'price','ma_5', 'ma_10', 'ma_30', 'ma_60', 'ma_120', 'ma_240', 'dema_5', 'dema_10', 'dema_30', 'dema_60', 'dema_120', 'dema_240', 'ema_5', 'ema_10', 'ema_30', 'ema_60', 'ema_120', 'ema_240', 'rsi_14', 'adx_14', 'natr_14', 'obv', 'type_price', 'avg_price', 'weighted_price', 'med_price', 'ht_dc_period', 'ht_trend_mode', 'beta_5', 'correl_30', 'linear_reg_5', 'linear_reg_10', 'linear_reg_22', 'stddev_5', 'stddev_10', 'stddev_22', 'tsf_5', 'tsf_10', 'tsf_22', 'var_5', 'var_10', 'var_22', 'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS', 'inphase', 'quadrature', 'upperband', 'middleband', 'lowerband', 'macd', 'macdsignal', 'macdhist']
train_label = "y_02_107"

normlize_features = [
    'open', 'high', 'low', 'price','ma_5', 'ma_10', 'ma_30', 'ma_60', 'ma_120', 'ma_240', 'dema_5', 'dema_10', 'dema_30', 'dema_60', 'dema_120', 'dema_240', 'ema_5', 'ema_10', 'ema_30', 'ema_60', 'ema_120', 'ema_240', 'type_price', 'avg_price', 'weighted_price', 'med_price', 'linear_reg_5', 'linear_reg_10', 'linear_reg_22', 'tsf_5', 'tsf_10', 'tsf_22',
]

def three_days_increase_retr(i, open, close, high, low, price, turn):
    if close[i] / close[i-1] >= 1.095 and close[i+1] / close[i] < 0.95: return True
    return False
    # if not (close[i] >  close[i-2]): return False
    # if not (turn[i] > turn[i-1] > turn[i-2]): return False
    # if turn[i] / turn[i-2] >= 2: return False
    # if turn[i] > 15: return False
    # if close[i] / close[i-i] >1.08: return False
    # return True
    
    
def calc_one(path):
    data = joblib.load(path)
    data = data[data["volume"] != 0]
    if len(data) <= 200: return 
    x, y = [], []
    train_data = data[train_features]
    date = train_data.index
    close = train_data.close.values
    high = train_data.high.values
    open = train_data.open.values
    low = train_data.low.values
    price = train_data.price.values
    volume = train_data.volume.values
    amount = train_data.amount.values
    turn = train_data.turn.values
    
    up, nday, ratio = explain_label(train_label)
    train_labels = data[train_label].values
    
    for i in range(seq_len, len(train_data)-nday):
        if high[i] / low[i] == 1: continue
        if not three_days_increase_retr(i, open, close, high, low, price, turn): continue
        x.append(list(train_data.iloc[i,:].values)+list(train_data.iloc[i,:].values-train_data.iloc[i-seq_len,:].values)+list(np.mean(train_data.iloc[i-seq_len:i,:], axis=0)))
        y.append(train_labels[i])
    return [x, y, np.random.random() < train_val_split]


def generate():
    trade_days, last_trade_day = get_last_update_date()
    
    train_data_x = []
    val_data_x = []
    train_data_y = []
    val_data_y = []
    paths = []
    for file in tqdm(os.listdir(DAILY_DIR)):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        paths.append(path)
       
    np.random.shuffle(paths)
    
    pbar = tqdm(total=len(paths))
    
    queue = multiprocessing.Queue(THREAD_NUM)
    queue_res = multiprocessing.Queue()
    
    def calc_ex_list(queue, queue_res):
        train_data_x = []
        val_data_x = []
        train_data_y = []
        val_data_y = []
        while True:
            argv = queue.get()
            if argv is None:
                queue_res.put(cPickle.dumps([train_data_x, train_data_y, val_data_x, val_data_y]))
                break
            instance = calc_one(argv)
            if instance is not None:
                if instance[2]:
                    train_data_x.extend(instance[0])
                    train_data_y.extend(instance[1])
                else:
                    val_data_x.extend(instance[0])
                    val_data_y.extend(instance[1])
                
    processes = [Process(target=calc_ex_list, args=(queue, queue_res, )) for _ in range(THREAD_NUM)]
    for each in processes:
        each.start()
    for path in paths:
        assert file is not None
        queue.put(path)
        pbar.update(1)

    # necessary because queue is out-of-order
    while not queue.empty():
        pass
    
    for i in range(THREAD_NUM):
        queue.put(None)

    pbar.close()

    train_data_x = []
    val_data_x = []
    train_data_y = []
    val_data_y = []

    pbar = tqdm(total=THREAD_NUM)
    for i in range(THREAD_NUM):
        t = queue_res.get()
        t = cPickle.loads(t)
        if t is not None:
            train_data_x.extend(t[0])
            train_data_y.extend(t[1])
            val_data_x.extend(t[2])
            val_data_y.extend(t[3])
        pbar.update(1)
    pbar.close()

    for each in processes:
        each.join()
            
    train_data_x = np.array(train_data_x)
    val_data_x = np.array(val_data_x)
    train_data_y = np.array(train_data_y)
    val_data_y = np.array(val_data_y)
    train_data_x = np.nan_to_num(train_data_x, 0)
    val_data_x = np.nan_to_num(val_data_x, 0)
    from collections import Counter
    print(Counter(train_data_y))
    np.savez("three_days_increase_train_lgb.npz", x=train_data_x, y=train_data_y)
    np.savez("three_days_increase_val_lgb.npz", x=val_data_x, y=val_data_y)
    print(train_data_x.shape)
    print(val_data_x.shape)

            
if __name__ == "__main__":
    generate()
    
    