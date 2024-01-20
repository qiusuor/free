from data.fetch_core import fetch
from utils import *
import platform
from data.check_data import check_daily
from data.discard import discard_labels

def fetch_daily_no_adjust():
    return fetch(freqs=['d'], num_thread=8, adjustflag='3')

if __name__ == "__main__":
    code_num = fetch_daily_no_adjust()
    if not code_num: exit(-1)
    if "Windows" in platform.platform():
        upload_data(local_paths=[r'C:\Users\qiusuo\Desktop\free\data\data\daily_download_no_adjust'], target_paths=["/home/qiusuo/free/data/data/daily_download_no_adjust/"])
    