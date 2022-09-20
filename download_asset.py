import sys
import os
import ccxt
import joblib
from crypto_data_fetcher.binance_spot import BinanceSpotFetcher


instrument = str(sys.argv[1]).upper()
timeframe  = int(sys.argv[2])


binance = ccxt.binance()
fetcher = BinanceSpotFetcher(ccxt_client=binance)

df = fetcher.fetch_ohlcv(
        market       = instrument,
        interval_sec = timeframe*60
        )

print(f"downloading {instrument} {timeframe}m, please wait... ")
print(df)

os.makedirs(f"./data/{instrument}", exist_ok=True)

df.to_csv(f"./data/{instrument}/{instrument}-{timeframe}m.csv", index=True, header=True)

print(f"saved to ./data/{instrument}/{instrument}-{timeframe}m.csv")