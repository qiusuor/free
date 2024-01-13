# coding=utf8

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
from sklearn import preprocessing
from collections import defaultdict
from tqdm import tqdm
import shutil
import os
import paramiko
import _pickle as cPickle
import hashlib
import inspect

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
    
def to_date(date):
    if isinstance(date, str) and "-" in date:
        return datetime.datetime.strptime(str(date), "%Y-%m-%d")
    return datetime.datetime.strptime(str(date), "%Y%m%d")

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

def render_html(data, title, html_path):
    line = Line(init_opts=opts.InitOpts(width="1450px",
                                    height="650px",
                                    page_title=title))
    line = line.add_xaxis(xaxis_data=data.index)
    for col in data.columns:
        if col == "date": continue
        line = line.add_yaxis(
            series_name=col,
            y_axis=data[col],
            yaxis_index=0,
            is_smooth=False,
            is_symbol_show=False,
        )
    line = line.set_global_opts(
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
    line.render(html_path)


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


def md5_encode(input_string):
    md5 = hashlib.md5()
    md5.update(input_string.encode('utf-8'))
    md5_digest = md5.hexdigest()
    return md5_digest

def pandas_rolling_agg(ref=None):
    def rolling(func):
        def agg(df_w):
            dfi =  ref[(ref.index >= df_w.index[0]) & (ref.index <= df_w.index[-1])]
            return func(dfi) 
        return agg
    return rolling

def hard_disk_cache(force_update=False):
    def get_result(func):
        md5 = md5_encode(inspect.getsource(func))
        cache_name = os.path.join(HARD_DISK_CACHE_DIR, func.__name__ + "_{}.pkl".format(md5))
        make_dir(cache_name)
        if force_update:
            if os.path.exists(cache_name):
                shutil.rmtree(cache_name)
        def wrapper(*args, **kw):
            if os.path.exists(cache_name):
                with open(cache_name, "rb") as f:
                    return cPickle.load(f)
            else:
                value = func(*args, **kw)
                with open(cache_name, "wb") as f:
                    cPickle.dump(value, f)
                return value
        return wrapper
    return get_result

def get_feature_cols():
    paths = main_board_stocks()
    df = joblib.load(paths[-5])
    no_feature_cols = set(["code", "open", "high", "low", "close", "preclose", "adjustflag", "tradestatus", "code_name", 'isST'] + [col for col in df.columns if col.startswith("y") or col.startswith("dy")])
    feature_cols = [col for col in df.columns if col not in no_feature_cols]
    assert len(feature_cols) > 0
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

def main_board_stocks():
    paths = []
    for file in os.listdir(DAILY_DIR):
        code = file.split("_")[0]
        if not_concern(code) or is_index(code):
            continue
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(DAILY_DIR, file)
        paths.append(path)
    return paths
