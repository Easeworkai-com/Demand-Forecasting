import pandas as pd

def load_data():
    return pd.read_csv("data/stock.csv", parse_dates=["Date"])

df1 = load_data()