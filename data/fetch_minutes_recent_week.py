from data.fetch_core import fetch
from config import *
from utils import *

def fetch_minutes_recent_week():
    trade_days = get_trade_days(update=False)
    fetch(freqs=['5'], start_date=to_str_date(trade_days[-5]), save_dir=MINUTE_RECENT_DIR)

if __name__ == "__main__":
    fetch_minutes_recent_week()
    