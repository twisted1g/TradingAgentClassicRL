import pandas as pd

df = pd.read_pickle("./data/binance-BTCUSDT-1h_2021.pkl")

df.to_csv("./data/data_1h_2021.csv", index=False)
