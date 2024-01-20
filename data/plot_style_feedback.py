from utils import *
from ydata_profiling import ProfileReport


df = joblib.load("/home/qiusuo/free/data/data/market/style_features.pkl")

# print(df.describe())
# print(df.columns)
# exit(0)
fileds = ["limit_up", "limit_up_1d", "limit_up_2d", "limit_up_3d", "limit_up_4d", "limit_up_5d", "limit_up_6d", "limit_up_7d", "limit_up_8d"]

for filed in fileds:
    keys = ["style_feat_shif1_of_y_next_1d_ret_mean_"+filed, "style_feat_shif1_of_y_next_1d_ret_std_"+filed]
    render_html(df[keys], filed, "{}.html".format(filed))

groups = {
    "limit_up": [col for col in df.columns if "limit_up" in col and not "limit_up_line" in col],
    "limit_up_line": [col for col in df.columns if "limit_up_line" in col],
    "limit_down": [col for col in df.columns if "limit_down" in col and not "limit_down_line" in col],
    "limit_down_line": [col for col in df.columns if "limit_down_line" in col],
    "high_price": [col for col in df.columns if "high_price" in col],
    "high_turn": [col for col in df.columns if "high_turn" in col],
}
for name, group in groups.items():
    profile = ProfileReport(df[group], title="Profiling Report")
    profile.to_file("style_feedback_{}.html".format(name))