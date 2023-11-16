from data.fetch_core import fetch
from utils import *

def fetch_daily():
    return fetch(freqs=['d'], num_thread=8)

if __name__ == "__main__":
    code_num = fetch_daily()
    if not code_num: exit(0)
    upload_data()
    