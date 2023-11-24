from data.fetch_core import fetch
from utils import *
import platform
from check_data import check_daily

def fetch_daily():
    return fetch(freqs=['d'], num_thread=8)

if __name__ == "__main__":
    code_num = fetch_daily()
    if not code_num: exit(-1)
    check_daily()
    if "Windows" in platform.platform():
        upload_data()
    