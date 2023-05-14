import os
from joblib import dump
import baostock as bs
import pandas as pd
import bisect
import tqdm
from config import DAILY_DIR, ALL_STOCKS
from utils import is_index, not_concern, fetch_stock_codes, make_dir

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

    if adjustflag=='1':
        result['factor'] = result['factor'].apply(factor).astype("float")
        result["volume"] = result["volume"] / result['factor']
    else:
        result['factor'] = result['factor'].apply(factor).astype("float")
        result["volume"] = result["volume"] / result['factor']


def update_one(code, login=False, frequency="d", adjustflag="1"):
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
    if os.path.exists(data_path):
        result=pd.read_csv(data_path)
        dealTime(result)
        start_date=result.index[-1].strftime("%Y-%m-%d")
        result=result[result.index<start_date]
    else:
        result = pd.DataFrame([], columns=list("date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST,factor".split(",")))
        dealTime(result)
        start_date = "1900-01-01"

    rs = bs.query_history_k_data_plus(code, fields,
                                 start_date=start_date, frequency=frequency, adjustflag=adjustflag)
    assert rs.error_code != 0, "fetch {} filed!".format(code)
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    if(len(data_list)<=0):return

    result_tmp = pd.DataFrame(data_list, columns=rs.fields)
    result_tmp.replace("", "0", inplace=True)
    dealTime(result_tmp)

    rs_list = []
    rs_factor = bs.query_adjust_factor(code=code, start_date="1900-01-01")
    while (rs_factor.error_code == '0') & rs_factor.next():
        rs_list.append(rs_factor.get_row_data())
    result_factor = pd.DataFrame(rs_list, columns=rs_factor.fields)
    dealTime(result_factor)

    if not is_index(code):
        calRehab(result_tmp, result_factor, adjustflag=adjustflag)
        
    if len(result_tmp)>0:
        result = pd.concat([result, result_tmp], axis=0)
        
    for col in list("open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST,factor".split(",")):
        result[col] = pd.to_numeric(result[col])
    make_dir(data_path)
    result.to_csv(data_path)

    if not login:
        bs.logout()

    dump(result, os.path.join(DAILY_DIR, "{}_{}_{}.pkl".format(code, frequency, adjustflag)))
    return result


def update(adjustflag='2', freqs=['m', 'w', 'd', '60', '30', '15', '5'], code_list=[]):
    fetch_stock_codes()
    lg = bs.login()
    assert lg.error_code != 0, "Login filed!"
    stockes = pd.read_csv(ALL_STOCKS)

    for freq in freqs:
        for code in tqdm.tqdm(code_list or stockes.code):
            if not_concern(code): continue
            print(code)
            update_one(code, login=True, frequency=freq, adjustflag=adjustflag)
    bs.logout()


def update_daily():
    update(freqs=['d'])


if __name__ == "__main__":
    update_daily()
    