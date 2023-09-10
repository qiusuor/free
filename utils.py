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
from tqdm import tqdm


def addGaussianNoise(x, std=0.05):
    return x + torch.normal(x, std)

def make_dir(file_name):
    if "." in os.path.basename(file_name):
        dir_name = os.path.dirname(file_name)
    else:
        dir_name = file_name
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
def not_concern(code):
    if code.startswith("sh.000001"):
        return False
    if code.startswith("sh.00"):
        return True
    if code.startswith("sz.39"):
        return True
    if code.startswith("sz.30"):
        return True
    if code.startswith("sh.68"):
        return True
    if code.startswith("bj"):
        return True
    return False


def is_index(code):
    if code.startswith("sh.00"):
        return True
    if code.startswith("sz.39"):
        return True
    return False


def get_trade_days(login=False):
    if not login:
        lg = bs.login()
        assert lg.error_code != 0, "Login failed!"
    rs = bs.query_trade_dates(start_date="2012-01-01")
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
        get_trade_days(login=login)
    trade_days = joblib.load(TRADE_DAYS_PKL)
    last_trade_day = trade_days[-1]
    return last_trade_day


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
        if high[j] / close[i+1] > expect_gain: return 1
    return 0

def get_down_label(i, open, close, high, low, price, turn, hold_day=2, tolerent_pay=0.97):
    for j in range(i+2, i+hold_day+1):
        if low[j] / close[i+1] < tolerent_pay: return 1
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
        no_feature_cols = set(["code", "adjustflag", "tradestatus"] + [col for col in df.columns if col.startswith("y") or col.startswith("dy")])
        feature_cols = [col for col in df.columns if col not in no_feature_cols]
        return feature_cols
    
def get_industry_info():
    lg = bs.login()

    rs = bs.query_stock_industry()

    industry_list = []
    while (rs.error_code == '0') & rs.next():
        industry_list.append(rs.get_row_data())
    result = pd.DataFrame(industry_list, columns=rs.fields)
    ind = dict()
    ind_mapping = dict()
    for updateDate,code,code_name,industry,industryClassification in result.values:
        if industry not in ind_mapping:
            ind_mapping[industry] = len(ind_mapping)
        ind[code] = ind_mapping[industry]
    joblib.dump(ind, INDUSTRY_INFO)

    