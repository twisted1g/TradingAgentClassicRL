import pandas as pd

df = pd.read_pickle("./data/binance-BTCUSDT-1m_2023.pkl")

df.to_csv("./data/data_1m_2023.csv", index=False)
