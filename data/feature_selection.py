from tsfresh.feature_selection.relevance import calculate_relevance_table
from joblib import load
from utils import *
from sklearn.preprocessing import LabelEncoder


df = load("feat.pkl")
# df = df[:to_date(20231110)]
df = df.fillna(0)
df = df.sample(frac=0.1)
# df["code_seq"] = LabelEncoder().fit_transform(df["code_name"])
# df["date_seq"] = LabelEncoder().fit_transform(df.index)

X = df[get_feature_cols()]
Y = df["y_02_109"]
mean, std = X.mean(0), X.std(0)
X = (X - mean) / (std + 1e-9)
relevance_table = calculate_relevance_table(X, Y)
print(relevance_table)
relevance_table.to_csv("relevance_table.csv", index=False)

relevance_table = pd.read_csv("relevance_table.csv")
useful_features = list(relevance_table[relevance_table.relevant].feature)
print(useful_features)

no_features = list(relevance_table[relevance_table.relevant == False].feature)
print(no_features)