# coding=utf8

from itertools import combinations
import numpy as np
import pandas as pd
import baostock as bs
from config import *
import joblib
import math
import bisect
import pyecharts.options as opts
from pyecharts.charts import Line
import datetime
import torch
from sklearn import preprocessing
from collections import defaultdict
from multiprocessing import Pool
from joblib import dump
from tqdm import tqdm
import shutil
import os
import paramiko

def make_dir(file_name):
    if "." in os.path.basename(file_name):
        dir_name = os.path.dirname(file_name)
    else:
        dir_name = file_name
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
def remove_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
        
def not_concern(code):
    if code.startswith("sh.000001"):
        return False
    return not (code.startswith("sh.60") or code.startswith("sz.00"))


def is_index(code):
    if code.startswith("sh.00"):
        return True
    if code.startswith("sz.39"):
        return True
    return False


def get_trade_days(login=False, update=True):
    if update:
        if not login:
            lg = bs.login()
            assert lg.error_code != 0, "Login failed!"
        rs = bs.query_trade_dates(start_date="2020-01-01")
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        result = result[result.is_trading_day == '1']
        trade_days = result.calendar_date.values
        trade_days = [int(x.replace('-', '')) for x in trade_days]
        make_dir(TRADE_DAYS)
        result.to_csv(TRADE_DAYS, encoding="gbk", index=False)
        joblib.dump(trade_days, TRADE_DAYS_PKL)
        if not login:
            bs.logout()
        return trade_days
    return joblib.load(TRADE_DAYS_PKL)

def get_offset_trade_day(day, n):
    trade_days = joblib.load(TRADE_DAYS_PKL)
    day_index = bisect.bisect_left(trade_days, day)
    return trade_days[day_index+n]

def to_str_date(int_date):
    return datetime.datetime.strptime(str(int_date), "%Y%m%d").strftime("%Y-%m-%d")
    
def to_date(int_date):
    return datetime.datetime.strptime(str(int_date), "%Y%m%d")

def to_int_date(date):
    return int(date.strftime("%Y%m%d"))

def get_last_trade_day(login=False, update=True):
    if update:
        get_trade_days(login=login, update=update)
    trade_days = joblib.load(TRADE_DAYS_PKL)
    localtime = datetime.datetime.now()
    if localtime.hour < 18 and to_int_date(localtime) in trade_days:
        return trade_days[-2]
    return trade_days[-1]


def fetch_stock_codes():
    lg = bs.login()
    last_trade_day = to_str_date(get_last_trade_day(login=True))
    assert lg.error_code != 0, "Login failed!"
    rs = bs.query_all_stock(day=last_trade_day)
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    make_dir(ALL_STOCKS)
    result.to_csv(ALL_STOCKS, index=False)
    bs.logout()

def render_html(code, data, html_path):
    (
        Line(init_opts=opts.InitOpts(width="1450px",
                                     height="650px",
                                     page_title=str(code)))
        .add_xaxis(xaxis_data=[str(item[0]) for item in data])
        .add_yaxis(
            series_name="div",
            y_axis=[item[1] for item in data],
            yaxis_index=0,
            is_smooth=False,
            is_symbol_show=False,
        )
        .add_yaxis(
            series_name="div_diff",
            y_axis=[item[2] for item in data],
            yaxis_index=0,
            is_smooth=False,
            is_symbol_show=False,
        )
        .add_yaxis(
            series_name="div_ratio",
            y_axis=[item[3] for item in data],
            yaxis_index=0,
            is_smooth=False,
            is_symbol_show=False,
        )
        .add_yaxis(
            series_name="avg",
            y_axis=[item[4] for item in data],
            yaxis_index=0,
            is_smooth=False,
            is_symbol_show=False,
        )
        .add_yaxis(
            series_name="price",
            y_axis=[item[5] for item in data],
            yaxis_index=0,
            is_smooth=False,
            is_symbol_show=False,
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="{}".format(code)),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            datazoom_opts=[
                opts.DataZoomOpts(xaxis_index=0),
                opts.DataZoomOpts(type_="inside", xaxis_index=0),
            ],
            xaxis_opts=opts.AxisOpts(type_="category",
                                     is_scale=True,
                                     ),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                name_location="start",
                is_scale=False,
                axistick_opts=opts.AxisTickOpts(is_inside=False),
            )
        )
        .render(html_path)
    )


def get_up_label(i, open, close, high, low, price, turn, hold_day=2, expect_gain=1.07):
    for j in range(i+2, i+hold_day+1):
        if high[j] / open[i+1] > expect_gain: return 1
    return 0

def get_down_label(i, open, close, high, low, price, turn, hold_day=2, tolerent_pay=0.97):
    for j in range(i+2, i+hold_day+1):
        if low[j] / open[i+1] < tolerent_pay: return 1
    return 0

def get_labels(open, close, high, low, price, turn, hold_day=2, expect_gain=1.07, tolerent_pay=0.97, up=True):
    labels = []
    N = len(open)
    for i in range(N):
        if i + hold_day < N:
            if up:
                labels.append(get_up_label(i, open, close, high, low, price, turn, hold_day, expect_gain))
            else:
                labels.append(get_down_label(i, open, close, high, low, price, turn, hold_day, tolerent_pay))
        else:
            labels.append(0)
    return labels
        
def explain_label(label):
    label = label.split("_")
    up = label[0] == "y"
    nday = int(label[1])
    ratio = float(label[2]) / 100
    
    return up, nday, ratio


def pandas_rolling_agg(ref=None):
    def rolling(func):
        def agg(df_w):
            dfi =  ref[(ref.index >= df_w.index[0]) & (ref.index <= df_w.index[-1])]
            return func(dfi) 
        return agg
    return rolling

def get_feature_cols():
    for file in os.listdir(DAILY_DIR):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        df = joblib.load(path)
        no_feature_cols = set(["code", "adjustflag", "tradestatus", "code_name", "shiborON", "shibor1W", "shibor2W", "shibor3M", "shibor9M", "shibor1Y", "industry", 'CDLTHRUSTING', 'CDLDOJI', 'CDLHIKKAKE', 'CDLHARAMICROSS', 'CDLEVENINGSTAR', 'CDLSTICKSANDWICH', 'CDLDRAGONFLYDOJI', 'CDLDOJISTAR', 'CDL3BLACKCROWS', 'CDLSEPARATINGLINES', 'CDLTAKURI', 'CDLEVENINGDOJISTAR', 'CDLLADDERBOTTOM', 'CDLONNECK', 'CDLUNIQUE3RIVER', 'CDLIDENTICAL3CROWS', 'CDLXSIDEGAP3METHODS', 'CDLADVANCEBLOCK', 'CDLCOUNTERATTACK', 'CDLGAPSIDESIDEWHITE', 'CDL3LINESTRIKE', 'CDL3INSIDE', 'CDLTASUKIGAP', 'CDLHIKKAKEMOD', 'CDLTRISTAR', 'CDLRISEFALL3METHODS', 'CDLBREAKAWAY', 'CDLABANDONEDBABY', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLMATHOLD', 'CDLCONCEALBABYSWALL', 'CDL3STARSINSOUTH', 'isST'] + [col for col in df.columns if col.startswith("y") or col.startswith("dy") or col.startswith("price_div_chip_avg_") or col.startswith("turn_div_mean_turn_") or col.startswith("limit_")] + ['factor'])
        feature_cols = [col for col in df.columns if col not in no_feature_cols]
        return feature_cols
    
def get_industry_info():
    lg = bs.login()

    rs = bs.query_stock_industry()

    industry_list = []
    while (rs.error_code == '0') & rs.next():
        industry_list.append(rs.get_row_data())
    result = pd.DataFrame(industry_list, columns=rs.fields)
    result["industry"] = preprocessing.LabelEncoder().fit_transform(result["industry"])
    ind = defaultdict(dict)
    for updateDate,code,code_name,industry,industryClassification in result.values:
        ind[code]["industry"] = industry
        ind[code]["code_name"] = code_name
    joblib.dump(ind, INDUSTRY_INFO)
    result.set_index("code", inplace=True)
    result.to_csv(INDUSTRY_INFO.replace("pkl", "csv"))
    bs.logout()

    
def get_shibor():
    lg = bs.login()
    rs = bs.query_shibor_data(start_date="2020-01-01")

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    result['date'] = pd.to_datetime(result['date'])
    result.set_index("date", inplace=True)
    result.to_csv(SHIBOR_INFO.replace("pkl", "csv"))
    joblib.dump(result, SHIBOR_INFO)
    bs.logout()
    
    
def get_profit(code, login):
    if not login:
        lg = bs.login()

    profit_list = []
    quarter = (datetime.datetime.now().month - 1) // 3
    year = datetime.datetime.now().year
    if quarter == 0:
        quarter = 4
        year -= 1
    rs_profit = bs.query_profit_data(code=code, year=year, quarter=quarter)
    while (rs_profit.error_code == '0') & rs_profit.next():
        profit_list.append(rs_profit.get_row_data())
    result_profit = pd.DataFrame(profit_list, columns=rs_profit.fields)
    # print(result_profit)

    if not login:
        bs.logout()
    return result_profit

 
class SSHConnection(object):
 
    def __init__(self, host, port, username, pwd):
        self.host = host
        self.port = port
 
        self.username = username
        self.pwd = pwd
        self.__k = None
 
    def connect(self):
        transport = paramiko.Transport((self.host, self.port))
        transport.connect(username=self.username, password=self.pwd)
        self.__transport = transport
 
    def close(self):
        self.__transport.close()
 
    def upload(self, local_path, target_path):
        sftp = paramiko.SFTPClient.from_transport(self.__transport)
        sftp.put(local_path, target_path)
 
    def download(self, remote_path, local_path):
        sftp = paramiko.SFTPClient.from_transport(self.__transport)
        sftp.get(remote_path, local_path)
 
    def cmd(self, command):
        ssh = paramiko.SSHClient()
        ssh._transport = self.__transport
        # 执行命令
        stdin, stdout, stderr = ssh.exec_command(command)
        # 获取命令结果
        result = stdout.read()
        # print(str(result, encoding='utf-8'))
        return result
 
 
def upload_data():
    ssh = SSHConnection(host='192.168.137.13', port=22, username='qiusuo', pwd='bahksysdd')
    ssh.connect()
    local_paths = [r'C:\Users\qiusuo\Desktop\free\data\data\daily_download', r"C:\Users\qiusuo\Desktop\free\data\data\market"]
    target_paths = ["/home/qiusuo/free/data/data/daily_download/", "/home/qiusuo/free/data/data/market/"]
    for local_path, target_path in zip(local_paths, target_paths):
        ssh.cmd("rm -rf {}".format(target_path))
        ssh.cmd("mkdir {}".format(target_path))
        for filename in tqdm(os.listdir(local_path)):
            ssh.upload(os.path.join(local_path, filename), target_path + filename)
    ssh.close()

def floor2(x):
    if np.isnan(x): return 1e6
    return (math.floor(x * 100)) / 100
    
def ceil2(x):
    if np.isnan(x): return -1e6
    return (math.ceil(x * 100)) / 100
    
def is_limit_up(df):
    return (df.close >= (df.close.shift(1) * 1.1).apply(floor2))

def is_limit_down(df):
    return (df.close <= (df.close.shift(1) * 0.9).apply(ceil2))

def is_limit_up_line(df):
    return is_limit_up(df) & (df.high == df.low)

def not_limit_up_line(df):
    return ~is_limit_up_line(df)

def is_limit_down_line(df):
    return is_limit_down(df) & (df.high == df.low)

def not_limit_down_line(df):
    return ~is_limit_down_line(df)

def is_reach_limit(df):
    return is_limit_down(df) | is_limit_up(df)

def not_reach_limit(df):
    return ~is_reach_limit(df)

def is_limit_line(df):
    return is_limit_down_line(df) | is_limit_up_line(df)

def not_limit_line(df):
    return ~is_limit_line(df)

def reserve_n_last(index, n=1):
    index.iloc[-n:] = True
    return index.astype(bool)
