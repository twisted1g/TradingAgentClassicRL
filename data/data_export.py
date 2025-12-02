import pandas as pd

df = pd.read_pickle("./data/binance-BTCUSDT-1h_2025.pkl")

df.to_csv("./data/data_1h_2025.csv", index=False)
