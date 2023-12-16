from utils import *
from config import *
   
def update():
    stockes = pd.read_csv(ALL_STOCKS)
    for code in tqdm(stockes.code):
        if not_concern(code): continue