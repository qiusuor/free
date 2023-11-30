import _pickle as cPickle
from ydata_profiling import ProfileReport


def analyse():
    cols = ["open", "high", "low", "close", "price", "turn", "volume", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "y_next_2_d_ret"]
    df = cPickle.load(open("whole_data.pkl", 'rb'))
    # print(df[(df.y_next_2_d_ret > 0) & (df.y_next_2_d_ret < 0.74)])
    
    # exit(0)
    profile = ProfileReport(df[cols], title="Raw Data analyse")
    profile.to_file("raw_data_analyse.html")
    
if __name__ == "__main__":
    analyse()
    
    