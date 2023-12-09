from data.fetch_core import fetch
from utils import *
import platform
from data.check_data import check_daily
from data.discard import discard_labels
from data.inject_labels import inject_labels

def fetch_daily():
    return fetch(freqs=['d'], num_thread=8)

if __name__ == "__main__":
    code_num = fetch_daily()
    if not code_num: exit(-1)
    discard_labels()
    check_daily()
    inject_labels()
    if "Windows" in platform.platform():
        upload_data()
    