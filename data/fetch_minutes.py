from data.fetch_core import fetch


def fetch_minutes():
    fetch(freqs=['5'])

if __name__ == "__main__":
    fetch_minutes()
    