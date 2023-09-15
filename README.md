# synthetic-prices

    Synthetic financial equity generation using Monte-Carlo simulation
    



# Todos

    DONE Define drawdown percentage to be considered as ruined
    DONE Generate Monte Carlo paths and calculate probability of risk of ruining
    DONE Extract Probabilistic Sharpe Ratio from each simulation
    DONE Extract Deflated Sharpe Ratio from all simulation
    DONE Generate 1 and 2 sigma probability cones for some confidence threshold




# steps to install some stuffs

    virtualenv -p python3.9 env && source env/bin/activate
    pip install -r requirements.txt
    pip install "git+https://github.com/richmanbtc/crypto_data_fetcher.git@v0.0.18#egg=crypto_data_fetcher"
    pip install TA-lib --no-binary TA-lib


# run some streamlit based stuffs

    streamlit run monte_carlo_equity_streamlit_single.py



# Resources and references

    - How do traders use Monte-Carlo simulations
        https://www.youtube.com/watch?v=yihYxhYp-DM
    
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



# Screenshots

![Screenshot of MC Equity](https://raw.githubusercontent.com/sharavsambuu/synthetic-prices/main/pictures/mc_equity_03.png)
