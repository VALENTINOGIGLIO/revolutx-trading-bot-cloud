import sqlite3
import pandas as pd
import sys

conn = sqlite3.connect("data/market_data.db")
df = pd.read_sql_query("SELECT * FROM candles WHERE symbol='DOGEEUR' AND interval='4h' ORDER BY open_time ASC", conn)

print("Rows:", len(df))

# Compute EMA 200
df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

print(df[['open_time', 'close', 'ema_200']].tail(20))
print("NaN count in ema_200:", df['ema_200'].isna().sum())

