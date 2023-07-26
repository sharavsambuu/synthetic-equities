# synthetic-prices

    Synthetic financial time series generation for the Monte-Carlo simulation
    https://synthetic-prices-q8eu1icg3.streamlit.app/ 



# Todos

    DONE Define drawdown percentage to be considered as ruined
    DONE Generate Monte Carlo paths and calculate probability of risk of ruining
    DONE Extract Probabilistic Sharpe Ratio from each simulation
    DONE Extract Deflated Sharpe Ratio from all simulation
    DONE Generate 1 and 2 sigma probability cones for some confidence threshold


    Implement Geometric Brownian Motion as baseline, 30minute timeframe from 1minute intrady 
    Implement Stochastic Volatility Models to tackle one of the shortcomings of GBM, allows
    to model time varying feature of volatility of the prices
        - implement Heston
        - implement SABR
    Implement Jump Diffusion Model for modelling sudden events and news
        - Merton Jump Diffusion
        - Kou Jump Diffusion
    Generate backtestable multiple scenarios from selected models
    Implement simple trading rule based on RSI or other indicators
        - backtest on multiple MC paths
        - Extract PSR, DSR, SR metrics from MC paths


    



# steps to install some stuffs

    virtualenv -p python3.9 env && source env/bin/activate
    pip install -r requirements.txt
    pip install "git+https://github.com/richmanbtc/crypto_data_fetcher.git@v0.0.18#egg=crypto_data_fetcher"
    pip install TA-lib --no-binary TA-lib


# run some streamlit based stuffs

    streamlit run monte_carlo_equity_streamlit_single.py


# Download historical data and generate dollar bars

    python download_asset.py btcusdt 1
    python generate_dollar_bar.py btcusdt 120000000


# Resources and references

    - Andrea Unger podcast
        https://open.spotify.com/episode/5EoE2rPlJs8pIxsnEh1BRb

    - Improving Your Algo Trading By Using Monte Carlo Simulation and Probability Cones
        https://kjtradingsystems.com/monte-carlo-probability-cones.html 

    - How to detect when a strategy is failing - Kevin Davey, Better System Trader
        https://www.youtube.com/watch?v=_dMbZfALIAQ

    - Probability cones related references
        How to use Probability Cones indicator in TradingView
        https://www.youtube.com/watch?v=sKTjxkYA3UU
        https://www.tradingview.com/script/Cp49eQt5-Probability-Cones/
        https://www.bausbenchmarks.com/indicators/probability-cones

    - Probabilistic Sharpe Ratio
        https://quantdare.com/probabilistic-sharpe-ratio/
        https://portfoliooptimizer.io/blog/the-probabilistic-sharpe-ratio-hypothesis-testing-and-minimum-track-record-length-for-the-difference-of-sharpe-ratios/
        https://stockviz.biz/2020/05/23/probabilistic-sharpe-ratio/
        https://www.youtube.com/watch?v=yKPNjru2Ry4

    - Deflated Sharpe Ratio
        https://quantdare.com/deflated-sharpe-ratio-how-to-avoid-been-fooled-by-randomness/
        https://github.com/rubenbriones/Probabilistic-Sharpe-Ratio

    - https://github.com/ydataai/ydata-synthetic

    - https://www.investopedia.com/articles/07/montecarlo.asp

    - https://towardsdatascience.com/simulating-stock-prices-in-python-using-geometric-brownian-motion-8dfd6e8c6b18 

    - https://stackoverflow.com/questions/53386933/how-to-solve-fit-a-geometric-brownian-motion-process-in-python

    - https://colab.research.google.com/github/ydataai/ydata-synthetic/blob/master/examples/timeseries/TimeGAN_Synthetic_stock_data.ipynb



# Screenshots

![Screenshot of MC Equity](https://raw.githubusercontent.com/sharavsambuu/synthetic-prices/main/pictures/mc_equity_03.png)
