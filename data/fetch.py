import os
from joblib import dump
import baostock as bs
import pandas as pd
import bisect
from config import DAILY_DIR, ALL_STOCKS
from utils import is_index, not_concern, fetch_stock_codes, make_dir
from multiprocessing import Pool
import threading
import time
import numpy as np
import shutil
from utils import *

"""
adjustflag:
    1: forward
    2: backward
    3: raw
"""


def dealTime(result):
    if 'amount' in result.columns:
        result['amount']=result['amount'].astype('float')
    if 'volume' in result.columns:
        result['volume']=result['volume'].astype('float')
    if "dividOperateDate" in result.columns:
        result.rename(columns={'dividOperateDate': 'date'}, inplace=True)
    if "time" in result.columns:
        result.drop("date", inplace=True, axis=1)
        result['date'] = pd.to_datetime(result['time'], format="%Y%m%d%H%M%S000")
        result.drop("time", inplace=True, axis=1)
        result.set_index("date", inplace=True)
    else:
        result['date'] = pd.to_datetime(result['date'])
        result.set_index("date", inplace=True)


def calRehab(result, result_factor, adjustflag):
    result["volume"]=result["volume"].apply(lambda x: float(x) if x else 0.0)
    if adjustflag=='3':return
    result['factor']=result.index.values

    result_factor_date=result_factor.index.values
    result_factor_factor=result_factor.backAdjustFactor if adjustflag=='1' else result_factor.foreAdjustFactor
    def factor(x):
        idx=bisect.bisect_right(result_factor_date, x)
        if idx >= len(result_factor_date):
            return result_factor_factor[-1]
        return result_factor_factor[idx-1]

    result['factor'] = result['factor'].apply(factor).astype("float")
    result["volume"] = result["volume"] / result['factor']


class RetThread(threading.Thread):
    def __init__(self, func, args=()):
        super(RetThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.result = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self) 
        try:
            return self.result
        except Exception:
            raise TimeoutError


def fetch_one_wrapper(argv):
    try:
        t = RetThread(func=fetch_one, args=argv)
        t.daemon =True
        t.start()
        t.join(10) 
        if t.is_alive() or t.result != 0:
            raise TimeoutError
    except:
        fetch_one_wrapper(argv)

def fetch_one(code, login, frequency, adjustflag):
    try:
        if not login:
            lg=bs.login()
            assert lg.error_code != 0, "Login filed!"

        data_path = os.path.join(DAILY_DIR, "{}_{}_{}.csv".format(code, frequency, adjustflag))

        data_list = []
        fields_dict = {
            'd': "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
            'w': "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg",
            'm': "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg",
        }
        fields = fields_dict.get(frequency, "date,time,code,open,high,low,close,volume,amount,adjustflag")
        start_date = "2020-01-01"
        rs = bs.query_history_k_data_plus(code, fields,
                                    start_date=start_date, frequency=frequency, adjustflag=adjustflag)
        assert rs.error_code != 0, "fetch {} filed!".format(code)
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        if(len(data_list)<=0):return
        
        result = np.array(data_list)
        result = np.unique(result, axis=0)
        result = pd.DataFrame(result, columns=rs.fields)
        result.replace("", "0", inplace=True)
        dealTime(result)

        rs_list = []
        rs_factor = bs.query_adjust_factor(code=code, start_date="1900-01-01")
        while (rs_factor.error_code == '0') & rs_factor.next():
            rs_list.append(rs_factor.get_row_data())
        result_factor = pd.DataFrame(rs_list, columns=rs_factor.fields)
            
        if not is_index(code):
            dealTime(result_factor)
            calRehab(result, result_factor, adjustflag=adjustflag)
        else:
            result["factor"] = 1.0
            
        for col in list("open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST,factor".split(",")):
            result[col] = pd.to_numeric(result[col])
        result["price"]=result["amount"]/(result['volume']+1e-9)
        make_dir(data_path)
        result = result[result["volume"] != 0]
        trade_days = get_trade_days(update=False)
        trade_day_inverse_mapping = dict(zip(trade_days, range(len(trade_days))))
        index = list(map(lambda x: trade_day_inverse_mapping[to_int_date(x)], list(result.index)))
        start_index = len(index) - 1
        while start_index >=1 and index[start_index] - index[start_index-1] < 30:
            start_index -= 1
        result = result.iloc[start_index:]
        
        result.to_csv(data_path)
        dump(result, os.path.join(DAILY_DIR, "{}_{}_{}.pkl".format(code, frequency, adjustflag)))
        if not login:
            bs.logout()
        return 0
    except:
        time.sleep(np.random.randint(10))
        return fetch_one(code, login, frequency, adjustflag)


def fetch(adjustflag='2', freqs=['m', 'w', 'd', '60', '30', '15', '5'], code_list=[]):
    fetch_stock_codes()
    get_industry_info()
    stockes = pd.read_csv(ALL_STOCKS)
    if os.path.exists(DAILY_DIR):
        shutil.rmtree(DAILY_DIR)
    code_list = []
    for freq in freqs:
        for code in tqdm(code_list or stockes.code):
            if not_concern(code): continue
            code_list.append([code, False, freq, adjustflag])
    # fetch_one("sz.000670", False, 'd', '2')
    pool = Pool(8)
    pool.imap_unordered(fetch_one_wrapper, code_list)
    pool.close()
    pool.join()
    
    print(len(code_list))

def fetch_daily():
    fetch(freqs=['d'])

if __name__ == "__main__":
    fetch_daily()
    