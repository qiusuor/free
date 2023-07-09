import os
from joblib import dump
import baostock as bs
import pandas as pd
import bisect
import tqdm
from config import DAILY_DIR, ALL_STOCKS
from utils import is_index, not_concern, fetch_stock_codes, make_dir
from multiprocessing import Pool

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

def fetch_one_wrapper(argv):
    try:
        fetch_one(argv)
    except:
        fetch_one_wrapper(argv)

def fetch_one(argv):
    code, login, frequency, adjustflag = argv
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

    result = pd.DataFrame(data_list, columns=rs.fields)
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
    result.to_csv(data_path)

    if not login:
        bs.logout()

    dump(result, os.path.join(DAILY_DIR, "{}_{}_{}.pkl".format(code, frequency, adjustflag)))
    return result


def fetch(adjustflag='2', freqs=['m', 'w', 'd', '60', '30', '15', '5'], code_list=[]):
    fetch_stock_codes()
    stockes = pd.read_csv(ALL_STOCKS)

    code_list = []
    for freq in freqs:
        for code in tqdm.tqdm(code_list or stockes.code):
            if not_concern(code): continue
            code_list.append([code, False, freq, adjustflag])
    pool = Pool(64)
    pool.map(fetch_one_wrapper, code_list)
    pool.close()
    pool.join()

def fetch_daily():
    fetch(freqs=['d'])

if __name__ == "__main__":
    fetch_daily()
    