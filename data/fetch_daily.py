from data.fetch_core import fetch

def fetch_daily():
    fetch(freqs=['d'])

if __name__ == "__main__":
    fetch_daily()
    