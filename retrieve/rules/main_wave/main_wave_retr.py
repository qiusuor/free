from config import *
import numpy as np
import os
import joblib
from utils import *
import math
import pyecharts.options as opts
from pyecharts.charts import Line
from multiprocessing import Pool
from pyecharts.globals import ThemeType
import bisect


def main_wave_retrieve(closes, chip_divs, i=-1, recent_days=500):
    if i >= 0:
        closes = closes[:i+1]
        chip_divs = chip_divs[:i+1]
    if len(closes) < recent_days:
        return False
    # if closes[-1] >= max(closes[-recent_days:-1]) and chip_divs[-1] >= max(chip_divs[-recent_days:-1]):
    if closes[-1] >= max(closes[-recent_days:-1]):
        return True
    return False


def plot_chip_div(argv):
    file, code, start_day, end_day, recent_days, slice_len, exp_decay_rate = argv
    path = os.path.join(DAILY_DIR, file)
    data = joblib.load(path)
    data = data[data.volume != 0]
    if data.index[-1] < end_day:
        return
    cur_days, chip_divs, chip_avgs, prices, opens, highs, lows, closes, indexs = calc_chip_div(
        data, code, start_day, end_day, slice_len, exp_decay_rate)

    retr_df = []
    for i, (cur_end_day, cur_chip_div, index) in enumerate(zip(cur_days, chip_divs, indexs)):
        if main_wave_retrieve(closes, chip_divs, i, recent_days):
            if cur_end_day >= to_date(20220601):
                closes_slice = data.close.iloc[:index]
                highest_day_index = closes_slice.argmax()
                retr_df.append([code, cur_end_day, cur_chip_div, closes[i], len(
                    closes_slice) - highest_day_index])

    if retr_df:
        retr_df = pd.DataFrame(data=retr_df, columns=[
            "code", "date", "chip_div", "close", "highest_day_dist"])
        retr_df.to_csv(os.path.join(REPORT_PATH_MAIN_WAVE_RETR,
                       "{}.csv".format(code)), index=False)
        if main_wave_retrieve(closes, chip_divs, -1, recent_days):
            retr_df.to_csv(os.path.join(
                TODAY_REPORT_PATH_MAIN_WAVE_RETR, "{}.csv".format(code)), index=False)

    chip_divs_diff = chip_divs[:]
    for i in range(1, len(chip_divs)):
        chip_divs_diff[i] = chip_divs[i] - chip_divs[i-1]

    chip_divs_ratio = chip_divs[:]
    for i in range(1, len(chip_divs)):
        chip_divs_ratio[i] = chip_divs[i] / chip_divs[i-1]

    gt_main_wave = set(pd.read_csv(GT_MAIN_WAVE).code.values)
    if code in gt_main_wave:
        html_path = os.path.join(
            HTML_PATH_GT_MAIN_WAVE_RETR, "{}.html".format(code))
    else:
        html_path = os.path.join(HTML_PATH_MAIN_WAVE_RETR,
                                 "{}.html".format(code))
    render_html(code, list(zip(cur_days, chip_divs, chip_divs_diff,
                chip_divs_ratio, chip_avgs, prices)), html_path)


def plot_chip_divs(start_day, end_day, recent_day, slice_len, exp_decay_rate):
    pool = Pool(THREAD_NUM)
    os.system("rm -rf {}/*.html".format(HTML_PATH_MAIN_WAVE_RETR))
    os.system("rm -rf {}/*.html".format(HTML_PATH_GT_MAIN_WAVE_RETR))
    os.system("rm -rf {}/*.csv".format(REPORT_PATH_MAIN_WAVE_RETR))
    os.system("rm -rf {}/*.csv".format(TODAY_REPORT_PATH_MAIN_WAVE_RETR))
    argvs = []
    for file in os.listdir(DAILY_DIR):
        code = file[:-4]
        if not_concern(code):
            continue
        if not file.endswith(".pkl"):
            continue
        argvs.append([file, code, start_day, end_day,
                     recent_day, slice_len, exp_decay_rate])
    pool.map(plot_chip_div, argvs)
    pool.close()
    pool.join()


def main_wave_retr():
    trade_days, last_trade_day = get_last_update_date()
    start_day, end_day, recent_day, slice_len, exp_decay_rate = to_date(20220101), last_trade_day, 250, 22, 0.997
    plot_chip_divs(start_day, end_day, recent_day,
                   slice_len, exp_decay_rate)
    

if __name__ == "__main__":
    main_wave_retr()
