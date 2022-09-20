# synthetic-prices

    Synthetic time series generation


# steps to install some stuffs

    virtualenv -p python3.9 env && source env/bin/activate
    pip install -r requirements.txt
    pip install "git+https://github.com/richmanbtc/crypto_data_fetcher.git@v0.0.18#egg=crypto_data_fetcher"
    pip install TA-lib --no-binary TA-lib


# Download historical data

    python download_asset.py btcusdt 1


# Resources and references

    - https://github.com/ydataai/ydata-synthetic
    - https://www.investopedia.com/articles/07/montecarlo.asp
    - https://towardsdatascience.com/simulating-stock-prices-in-python-using-geometric-brownian-motion-8dfd6e8c6b18 
