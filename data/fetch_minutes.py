from data.fetch_core import fetch
from config import *

def fetch_minutes():
    fetch(freqs=['5'], save_dir=MINUTE_DIR)

if __name__ == "__main__":
    fetch_minutes()
    