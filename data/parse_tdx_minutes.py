# coding: UTF-8
from struct import *
import datetime
import os
from multiprocessing import Pool
from config import *
from utils import *

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
    cols = ["date", "day", "code", "open", "high", "low", "close", "amount", "volume", "price"]
    data = []
    for i in range(int(no)):
        a = unpack('HHfffffII', buf[b:e])
        num=a[0]
        year = num // 2048 + 2004
        month = (num % 2048) // 100
        day = (num % 2048) %100
        dt=datetime.datetime(year=year, month=month, day=day, hour=a[1]//60, minute=a[1]%60)
        data.append([dt, dt.strftime("%Y-%m-%d"), fname[:2]+'.'+fname[2:-4], a[2], a[3], a[4], a[5], a[6], a[7], a[6]/(a[7]+1e-9)])
        b = b + 32
        e = e + 32
    df = pd.DataFrame(data, columns=cols)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    df.to_csv(targetDir + os.sep + fname[:2]+'.'+fname[2:-4]+'_'+freq+ '_2' + '.csv')
    joblib.dump(df, targetDir + os.sep + fname[:2]+'.'+fname[2:-4]+'_'+freq+ '_2' + '.pkl')


if __name__ == "__main__":
    pathdir1 = r'C:\new_tdx\vipdoc\sh\minline'
    pathdir2 = r'C:\new_tdx\vipdoc\sz\minline'

    targetDir = TDX_MINUTE_DIR
    make_dir(targetDir)
    

    argvs = [(os.path.join(pathdir1, f), targetDir) for f in os.listdir(pathdir1)]
    argvs += [(os.path.join(pathdir2, f), targetDir) for f in os.listdir(pathdir2)]
    day2csv_data(argvs[0])
    pool = Pool(THREAD_NUM)
    pool.imap_unordered(day2csv_data, argvs)
    pool.close()
    pool.join()


    