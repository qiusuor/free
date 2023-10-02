import os
from data.fetch_daily import fetch


def fetch_minutes():
    fetch(freqs=['5'])

if __name__ == "__main__":
    fetch_minutes()
    