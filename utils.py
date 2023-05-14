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


def make_dir(file_name):
    dir_name = os.path.dirname(file_name)
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


def to_str_date(int_date):
    return datetime.datetime.strptime(str(int_date), "%Y%m%d").strftime("%Y-%m-%d")
    
def get_last_update_date(login=False):
    get_trade_days(login=login)
    trade_days = joblib.load(TRADE_DAYS_PKL)
    last_trade_day = trade_days[-1]
    return trade_days, last_trade_day


def fetch_stock_codes():
    lg = bs.login()
    last_trade_day = to_str_date(get_last_update_date(login=True)[1])
    assert lg.error_code != 0, "Login failed!"
    rs = bs.query_all_stock(day=last_trade_day)
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    make_dir(ALL_STOCKS)
    result.to_csv(ALL_STOCKS, index=False)
    bs.logout()


def exp_decay(pre_i_day, decay_rate=0.99):
    return decay_rate ** pre_i_day

def guass_decay(pre_i_day, sigma=120):
    return math.exp(-pre_i_day * pre_i_day / sigma / sigma)

def get_decay_coef(n, rate=0.99, sigma=120, which="exp"):
    if which == "exp":
        return [exp_decay(n - i - 1, rate) for i in range(n)]
    elif which == "guass":
        return [guass_decay(n - i - 1, sigma) for i in range(n)]
    else:
        raise NotImplementedError


def calc_chip_div(data, code, start_day=20220101, end_day=20230317, slice_len=12, exp_decay_rate=0.996):
    decay_coef = get_decay_coef(
        slice_len, rate=exp_decay_rate, which="exp")

    cur_days = []
    prices = []
    chip_avgs = []
    chip_divs = []
    closes = []
    highs = []
    lows = []
    opens = []
    indexs = []

    date = data.index.values
    close = data.close.values
    high = data.high.values
    open = data.open.values
    low = data.low.values
    price = data.price.values
    vol = data.vol.values
    amount = data.amount.values

    start_index = bisect.bisect_left(date, start_day)
    start_index = max(start_index, slice_len)
    end_index = bisect.bisect_left(date, end_day)
    assert end_index < len(date), (date, code, end_index, len(date), end_day)

    for i in range(start_index, end_index + 1):
        amount_slice = amount[i-slice_len:i]
        amount_slice *= decay_coef
        vol_slice = vol[i-slice_len:i]
        vol_slice *= decay_coef
        sum_vol_slice = sum(vol_slice)
        price_slice = price[i-slice_len:i]
        chip_avg = sum(amount_slice) / sum_vol_slice
        chip_div = 0
        for p, v in zip(price_slice, vol_slice):
            chip_div += (1 - p / chip_avg) ** 2 * v / sum_vol_slice

        chip_avgs.append(chip_avg)
        chip_divs.append(chip_div * 1000)
        prices.append(price[i])
        cur_days.append(date[i])
        closes.append(close[i])
        highs.append(high[i])
        opens.append(open[i])
        lows.append(low[i])
        indexs.append(i)

    return cur_days, chip_divs, chip_avgs, prices, opens, highs, lows, closes, indexs

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

