import pandas as pd

df = pd.read_pickle("./data/binance-BTCUSDT-15m_2025.pkl")

df.to_csv("./data/data_15m_2025.csv", index=False)
