import os
from multiprocessing import cpu_count

THREAD_NUM = cpu_count()
SEARCH_END_DAY = 21990101
DATA_START_DAY = 20210101
if "SEARCH_END_DAY" in os.environ:
    SEARCH_END_DAY = int(os.environ["SEARCH_END_DAY"])
VAL_N_LAST_DAY = 5
TEST_N_LAST_DAY = 5
val_delay_day = 0

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data/data")
TICK_DIR = os.path.join(DATA_DIR, "tick")
DAILY_DOWLOAD_DIR = os.path.join(DATA_DIR, "daily_download")
DAILY_DOWLOAD_DIR_NO_ADJUST = os.path.join(DATA_DIR, "daily_download_no_adjust")
DAILY_DIR = os.path.join(DATA_DIR, "daily")
DAILY_BY_DATE_DIR = os.path.join(DATA_DIR, "daily_by_date")
MINUTE_DIR = os.path.join(DATA_DIR, "minutes")
MINUTE_DIR_TMP = os.path.join(DATA_DIR, "minutes_tmp")
MINUTE_FEAT = os.path.join(DATA_DIR, "minutes_features")

EXP_CLS_DIR = os.path.join(ROOT_DIR, "rank/exp_cls")
EXP_CLS_DATA_CACHE = os.path.join(EXP_CLS_DIR, "cache_{}.pkl")
EXP_CLS_PRED_DIR = os.path.join(ROOT_DIR, "rank/exp_cls_pred")

EXP_RANK_DIR = os.path.join(ROOT_DIR, "rank/exp_rank")
EXP_RANK_DATA_CACHE = os.path.join(EXP_RANK_DIR, "cache_{}.pkl")
EXP_RANK_PRED_DIR = os.path.join(ROOT_DIR, "rank/exp_rank_pred")

INDUSTRY_INFO = os.path.join(DATA_DIR, "market/industry.pkl")
TRADE_DAYS = os.path.join(DATA_DIR, "market/trade_days.csv")
TRADE_DAYS_PKL = os.path.join(DATA_DIR, "market/trade_days.pkl")
ALL_STOCKS = os.path.join(DATA_DIR, "market/all_stocks.csv")
STYLE_FEATS = os.path.join(DATA_DIR, "market/style_features.pkl")
COMPANY_INFO = os.path.join(DATA_DIR, "market/company_details.pkl")

HARD_DISK_CACHE_DIR = os.path.join(DATA_DIR, "cache")

SH_INDEX = os.path.join(DAILY_DOWLOAD_DIR, "sh.000001_d_2.pkl")

