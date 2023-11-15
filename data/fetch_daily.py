from data.fetch_core import fetch
from utils import *

def fetch_daily():
    fetch(freqs=['d'], num_thread=8)

if __name__ == "__main__":
    fetch_daily()
    upload_data()
    