import pandas as pd

df = pd.read_pickle("./data/binance-BTCUSDT-1m.pkl")

df.to_csv("./data/data_1m.csv", index=False)
