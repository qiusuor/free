import os
from multiprocessing import cpu_count

THREAD_NUM = cpu_count()
SEARCH_END_DAY = 21990101
if "SEARCH_END_DAY" in os.environ:
    SEARCH_END_DAY = int(os.environ["SEARCH_END_DAY"])
VAL_N_LAST_DAY = 5
TEST_N_LAST_DAY = 5
val_delay_day = 10

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data/data")
TICK_DIR = os.path.join(DATA_DIR, "tick")
DAILY_DOWLOAD_DIR = os.path.join(DATA_DIR, "daily_download")
DAILY_DIR = os.path.join(DATA_DIR, "daily")
DAILY_BY_DATE_DIR = os.path.join(DATA_DIR, "daily_by_date")
TDX_MINUTE_DIR = os.path.join(DATA_DIR, "tdx_minutes")
TDX_MINUTE_RECENT_DIR = os.path.join(DATA_DIR, "tdx_minutes_recent")
MINUTE_FEAT = os.path.join(DATA_DIR, "minutes_features")

EXP_CLS_DIR = os.path.join(ROOT_DIR, "rank/exp_cls")
EXP_CLS_DATA_CACHE = os.path.join(EXP_CLS_DIR, "cache_{}.pkl")
EXP_CLS_PRED_DIR = os.path.join(ROOT_DIR, "rank/exp_cls_pred")

EXP_RANK_DIR = os.path.join(ROOT_DIR, "rank/exp_rank")
EXP_RANK_DATA_CACHE = os.path.join(EXP_RANK_DIR, "cache_{}.pkl")
EXP_RANK_PRED_DIR = os.path.join(ROOT_DIR, "rank/exp_rank_pred")

INDUSTRY_INFO = os.path.join(DATA_DIR, "market/industry.pkl")
SHIBOR_INFO = os.path.join(DATA_DIR, "market/shibor.pkl")
TRADE_DAYS = os.path.join(DATA_DIR, "market/trade_days.csv")
TRADE_DAYS_PKL = os.path.join(DATA_DIR, "market/trade_days.pkl")
ALL_STOCKS = os.path.join(DATA_DIR, "market/all_stocks.csv")
GT_MAIN_WAVE = os.path.join(DATA_DIR, "market/gt_main_waves.csv")
STYLE_FEATS = os.path.join(DATA_DIR, "market/style_features.pkl")

HARD_DISK_CACHE_DIR = os.path.join(DATA_DIR, "cache")


# main wave 
HTML_PATH_MAIN_WAVE_RETR = os.path.join(
    ROOT_DIR, "retrieve/rules/main_wave/html_main_wave_retr")
HTML_PATH_GT_MAIN_WAVE_RETR = os.path.join(
    ROOT_DIR, "retrieve/rules/main_wave/html_main_wave_retr/positive")
REPORT_PATH_MAIN_WAVE_RETR = os.path.join(
    ROOT_DIR, "retrieve/rules/main_wave/report_main_wave_retr")
TODAY_REPORT_PATH_MAIN_WAVE_RETR = os.path.join(
    ROOT_DIR, "retrieve/rules/main_wave/today_report_main_wave_retr")


# second wave 
second_wave_min_trade_day = 200
second_wave_retr_within_main_wave_day = 22
second_wave_retr_delay_main_wave_day = 3
second_wave_retr_main_wave_ratio = 2.0
second_wave_retr_main_wave_climp_day = 30
second_wave_retr_continuous_climp_n_day = 2
second_wave_retr_main_wave_drawback_upper = 0.8
second_wave_retr_main_wave_drawback_lower = 0.5
second_wave_wait_day = 4
second_wave_target_upper = 1.08
second_wave_target_lower = 0.92
second_wave_retr_file = os.path.join(DATA_DIR, "retr/second_wave/second_wave_retr.pkl")
second_wave_retr_des_file = os.path.join(DATA_DIR, "retr/second_wave/second_wave_retr_describe.csv")
second_wave_feature_len = 80



labels = [
    "y_02_101",
    "y_02_103",
    "y_02_105",
    "y_02_107",
    "y_02_109",
    
    "dy_02_97",
    "dy_02_95",
    "dy_02_92",
]

auto_encoder_features = ["turn", "price", "open", "low", "high", "close", "pctChg"]


auto_encoder_config = [
    (1024, 500, 3, 8),
    (1024, 500, 5, 8),
    (1024, 500, 10, 16),
    (1024, 500, 22, 16),
    (1024, 500, 60, 32),
    (1024, 500, 120, 32),
    (1024, 500, 240, 32),
]