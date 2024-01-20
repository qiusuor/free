import numpy as np
import pandas as pd
import time
from bs4 import BeautifulSoup
import datetime
import requests
import re
import threading
import asyncio
from pyppeteer import launch
from pyppeteer_stealth import stealth
from multiprocessing import Pool
from utils import *
from joblib import dump, load

async def fetch_detail_core(code):
    
    browser = await launch(headless=True, args=["--disable-infobars"])
    context = await browser.createIncognitoBrowserContext()
    page = await context.newPage()
    await stealth(page)
    await page.evaluateOnNewDocument("() =>{ Object.defineProperties(navigator, { webdriver: { get(): () => false} }) }")
    time.sleep(np.random.randint(0, 3))
    # url = "http://q.10jqka.com.cn/gn/index/field/addtime/order/desc/page/" + str(i+1) + "/ajax/1/"
    url = f"https://stockpage.10jqka.com.cn/{code}/"
    
    await page.goto(url)
    data = await page.content()
    
    try:
        soup = BeautifulSoup(data, 'lxml').find_all("dl", "company_details")
        # print(soup, url)
        soup = soup[0].find_all(["dt", "dd"])
    except Exception as ex:
        print(code, str(ex), soup)
        return None, None
    # print(soup)
    company_details = dict()
    for i in range(0, len(soup)-1):
        # print(soup[i].text, soup[i+1].text)
        if "涉及概念" in soup[i].text:
            company_details["concepts"] = soup[i+1].get('title').split('，')
        elif "每股净资产" in soup[i].text:
            company_details["net_asset_per_share"] = float(soup[i+1].text[:-1])
        elif "每股收益" in soup[i].text:
            company_details["profit_per_share"] = float(soup[i+1].text[:-1])
        elif "净利润" in soup[i].text and not "净利润增长率" in soup[i].text:
            assert "亿" in soup[i+1].text, soup[i+1].text
            company_details["net_profit"] = float(soup[i+1].text[:-2])
        elif "净利润增长率" in soup[i].text:
            company_details["net_profit_growth_rate"] = float(soup[i+1].text[:-1])
        elif "营业收入" in soup[i].text:
            assert "亿" in soup[i+1].text
            company_details["income"] = float(soup[i+1].text[:-2])
        elif "每股现金流" in soup[i].text:
            company_details["cash_flow_per_share"] = float(soup[i+1].text[:-1])
        elif "每股公积金" in soup[i].text:
            company_details["provident_per_share"] = float(soup[i+1].text[:-1])
        elif "每股未分配利润" in soup[i].text:
            company_details["undistributed_earning_per_share"] = float(soup[i+1].text[:-1])
        elif "总股本" in soup[i].text:
            assert "亿" in soup[i+1].text 
            company_details["total_share_capital"] = float(soup[i+1].text[:-1])
        elif "流通股" in soup[i].text:
            assert "亿" in soup[i+1].text
            company_details["outstanding_shares"] = float(soup[i+1].text[:-1])
    print(code, company_details)
    await browser.close()
     
    return code, company_details 
    
def fetch_detail_wrapper(code):
    ret = asyncio.get_event_loop().run_until_complete(fetch_detail_core(code))
    return ret
    
    
def fetch_detail():
    if os.path.exists(COMPANY_INFO):
        company_info = load(COMPANY_INFO)
    else:
        company_info = dict()
    all_stocks = [x[-14:-8] for x in main_board_stocks() if x[-14:-8] not in company_info]
    np.random.shuffle(all_stocks)
    all_stocks = all_stocks
    pool = Pool(1)
    ret = pool.imap_unordered(fetch_detail_wrapper, all_stocks)
    pool.close()
    pool.join()
    header = ["code", "concepts", "net_asset_per_share", "profit_per_share", "net_profit", "net_profit_growth_rate", "income", "cash_flow_per_share", "provident_per_share", "undistributed_earning_per_share", "total_share_capital", "outstanding_shares"]
    data = []
    
    for code, info in ret:
        for filed in header:
            if filed == "code":
                row = [str(code)]
            else:
                row.append(info[filed])
        if code:
            company_info[code] = row
        
    for code, row in company_info.items():
        data.append(row)
    data = pd.DataFrame(data, columns=header)
    data.to_csv(COMPANY_INFO.replace("pkl", "csv"), index=False)
    dump(company_info, COMPANY_INFO)
    
        

if __name__ == "__main__":
    fetch_detail()
    