# coding: UTF-8
from struct import *
import datetime
import os
from multiprocessing import Pool
from config import *
from utils import *
import platform

def day2csv_data(argv):
    path, targetDir = argv
    freq = '1'
    fname = os.path.basename(path)
    ofile = open(path, 'rb')
    buf = ofile.read()
    ofile.close()

    num = len(buf)
    no = num / 32
    b = 0
    e = 32
    cols = ["date", "day", "open", "high", "low", "close", "volume"]
    data = []
    for i in range(int(no)):
        a = unpack('HHfffffII', buf[b:e])
        num=a[0]
        year = num // 2048 + 2004
        month = (num % 2048) // 100
        day = (num % 2048) %100
        dt=datetime.datetime(year=year, month=month, day=day, hour=a[1]//60, minute=a[1]%60)
        data.append([dt, to_date(dt.strftime("%Y-%m-%d")), a[2], a[3], a[4], a[5], a[7]])
        b = b + 32
        e = e + 32
    df = pd.DataFrame(data, columns=cols)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    path = targetDir + os.sep + fname[:2]+'.'+fname[2:-4]+'_'+freq+ '_3' + '.pkl'
    df.to_csv(path.replace("pkl", "csv"))
    joblib.dump(df, path)

def parse(targetDir):
    pathdir1 = r'C:\new_tdx\vipdoc\sh\\minline'
    pathdir2 = r'C:\new_tdx\vipdoc\sz\\minline'
    make_dir(targetDir)

    argvs = [(os.path.join(pathdir1, f), targetDir) for f in os.listdir(pathdir1)]
    argvs += [(os.path.join(pathdir2, f), targetDir) for f in os.listdir(pathdir2)]
    day2csv_data(argvs[0])
    pool = Pool(THREAD_NUM)
    pool.imap_unordered(day2csv_data, argvs)
    pool.close()
    pool.join()
    remove_dir(pathdir1)
    remove_dir(pathdir2)
    make_dir(pathdir1)
    make_dir(pathdir2)

def parse_recent():
    remove_dir(MINUTE_DIR_TMP)
    parse(MINUTE_DIR_TMP)

def merge_data_core(path):
    old_path = os.path.join(MINUTE_DIR, path)
    new_path = os.path.join(MINUTE_DIR_TMP, path)
    df = joblib.load(new_path)
    if os.path.exists(old_path):
        old_data = joblib.load(old_path)
        if "date" in old_data.columns:
            old_data.set_index("date", inplace=True)
        last_day = old_data.day.values[-1]
        df = pd.concat([old_data, df[df.day > last_day]])
    df.to_csv(old_path.replace("pkl", "csv"))
    joblib.dump(df, old_path)
        
def merge_data():
    paths = os.listdir(MINUTE_DIR_TMP)
    pool = Pool(64)
    pool.imap_unordered(merge_data_core, paths)
    pool.close()
    pool.join()
    
if __name__ == "__main__":
    if "Windows" in platform.platform():
        parse_recent()
        upload_data(local_paths=[r'C:\Users\qiusuo\Desktop\free\data\data\minutes_tmp'], target_paths=["/home/qiusuo/free/data/data/minutes_tmp/"])
    else:
        merge_data()
        remove_dir(MINUTE_DIR_TMP)
    


    