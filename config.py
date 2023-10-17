import os
from multiprocessing import cpu_count

THREAD_NUM = cpu_count()
SEARCH_END_DAY = 21990101
# SEARCH_END_DAY = 20231009
TEST_N_LAST_DAY = 5
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data/data")
TICK_DIR = os.path.join(DATA_DIR, "tick")
DAILY_DIR = os.path.join(DATA_DIR, "daily")
TDX_MINUTE_DIR = os.path.join(DATA_DIR, "tdx_minutes")
TDX_MINUTE_RECENT_DIR = os.path.join(DATA_DIR, "tdx_minutes_recent")
MINUTE_FEAT = os.path.join(DATA_DIR, "minutes_features")
EXP_DIR = os.path.join(ROOT_DIR, "rank/exp")
EXP_DATA_CACHE = os.path.join(EXP_DIR, "cache_{}.pkl")
EXP_PRED_DIR = os.path.join(ROOT_DIR, "rank/exp_pred")
INDUSTRY_INFO = os.path.join(DATA_DIR, "market/industry.pkl")
SHIBOR_INFO = os.path.join(DATA_DIR, "market/shibor.pkl")
TRADE_DAYS = os.path.join(DATA_DIR, "market/trade_days.csv")
TRADE_DAYS_PKL = os.path.join(DATA_DIR, "market/trade_days.pkl")
ALL_STOCKS = os.path.join(DATA_DIR, "market/all_stocks.csv")
GT_MAIN_WAVE = os.path.join(DATA_DIR, "market/gt_main_waves.csv")

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
    "y_02_1094",
    "y_03_105",
    "y_03_107",
    "y_03_109",
    "y_03_115",
    "y_05_108",
    "y_05_115",
    "y_05_120",
    "y_05_135",
    "y_07_108",
    "y_07_115",
    "y_07_120",
    "y_07_135",
    "y_07_150",
    "y_15_108",
    "y_15_115",
    "y_15_120",
    "y_15_130",
    "y_15_150",
    "y_30_108",
    "y_30_115",
    "y_30_125",
    "y_30_150",
    "y_30_175",
    "y_60_110",
    "y_60_120",
    "y_60_130",
    "y_60_140",
    "y_60_150",
    "y_60_175",
    "y_60_200",
    
    
    "dy_02_97",
    "dy_02_95",
    "dy_02_92",
    "dy_05_97",
    "dy_05_95",
    "dy_05_92",
    "dy_07_95",
    "dy_07_92",
    "dy_15_92",
    "dy_15_90",
    "dy_30_92",
    "dy_30_90",
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