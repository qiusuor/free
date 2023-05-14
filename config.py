import os

CHECK_DAY = "20990101"
# CODE_AT_DATE = "2023-05-10"


THREAD_NUM = 4

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data/data")
TICK_DIR = os.path.join(DATA_DIR, "tick")
DAILY_DIR = os.path.join(DATA_DIR, "daily")
FEAT_DIR = os.path.join(DATA_DIR, "features")

TRADE_DAYS = os.path.join(DATA_DIR, "market/trade_days.csv")
TRADE_DAYS_PKL = os.path.join(DATA_DIR, "market/trade_days.pkl")

ALL_STOCKS = os.path.join(DATA_DIR, "market/all_stocks.csv")



# GT_MAIN_WAVE = os.path.join(DATA_DIR, "market/main_waves.csv")
# HTML_PATH_MAIN_WAVE_RETR = os.path.join(
#     ROOT_DIR, "retrieve/main_wave/html_main_wave_retr")
# HTML_PATH_GT_MAIN_WAVE_RETR = os.path.join(
#     ROOT_DIR, "retrieve/main_wave/html_main_wave_retr/positive")
# REPORT_PATH_MAIN_WAVE_RETR = os.path.join(
#     ROOT_DIR, "retrieve/main_wave/report_main_wave_retr")
# TODAY_REPORT_PATH_MAIN_WAVE_RETR = os.path.join(
#     ROOT_DIR, "retrieve/main_wave/today_report_main_wave_retr")

# EPS = 1e-6

# LOOK_AHEAD = 250
# MAX_WINDOW = 300
# ALPHA_DENPENDENCY_LEN = 250


# LOOK_AHEAD_CHIP_FEAT = 500


# MAIN_WAVE_RETR_CHIP_DIV_THRESH = 5

# RAW_FEAT = [
#     "date", "code", "open", "high", "low", "close", "forward_just", "backward_just", "rate", "vol", "amount", "turn", "circulation_val", "total_value", "limit_up", "limit_down", "pe_ttm", "psr_ttm", "pcm_ttm", "pbr_ttm", "ma_5", "ma_10", "ma_20", "ma_30", "ma_60", "macd_dif", "macd_dea", "macd_macd", "macd_cross", "kdj_k", "kdj_d", "kdj_j", "kdj_cross", "boll", "boll_up", "boll_down", "psy", "psyma", "rsi1", "rsi2", "rsi3", "amp", "vol_rate", "forward_rate", "price", "ma_cross_0", "ma_cross_1", "ma_cross_2", "ma_cross_3", "ma_cross_4", "ma_cross_5", "ma_cross_6", "ma_cross_7", "ma_cross_8", "ma_cross_9"
# ]
