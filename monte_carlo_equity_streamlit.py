import multiprocessing
import streamlit         as st
import pandas            as pd
import numpy             as np
import quantstats        as qs
import matplotlib.pyplot as plt
from   datetime          import timedelta
from   backtesting       import Backtest, Strategy
from   multiprocessing   import Pool


def mc_equity(params):
    pct_changes = params[0]
    return None


def simulate_equity_mc(df, num_simulations):
    params = [(df['pct_change']) for _ in range(0, num_simulations)]
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    result_list = pool.map(mc_equity, params)
    pool.close()
    pool.join()
    print(result_list)
    pass


def generate_equity(start_date, end_date, num_trades, initial_cash):
    trade_dates = pd.to_datetime(np.sort(np.random.choice(pd.date_range(start=start_date, end=end_date, periods=num_trades + 1), num_trades, replace=False)))
    percentage_changes = np.random.uniform(-0.043, 0.045, num_trades)
    df = pd.DataFrame({
        'datetime'  : trade_dates,
        'pct_change': percentage_changes,
    })
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df['cumret' ] = df['pct_change'].cumsum()
    df['cash'] = (1+df['pct_change']).cumprod()*initial_cash

    fig, ax = plt.subplots()
    ax.plot(df['cash'])
    st.pyplot(fig)

    sharpe_ratio  = qs.stats.sharpe       (df['pct_change'])
    profit_factor = qs.stats.profit_factor(df['pct_change'])
    max_drawdown  = qs.stats.max_drawdown (df['pct_change'])

    stats_df = pd.DataFrame()
    stats_df['Equity'       ] = [round(df.iloc[-1]['cash'],0)]
    stats_df['Sharpe Ratio' ] = [sharpe_ratio ]
    stats_df['Profit Factor'] = [profit_factor]
    stats_df['Max Drawdown' ] = [max_drawdown ]
    st.dataframe(stats_df)

    col11, _, _ = st.columns(3)
    with col11:
        num_simulations = st.number_input("N Simulations", min_value=10, step=1, value=50)
    if st.button("MC Simulate Equity"):
        simulate_equity_mc(df, num_simulations)



def main():
    st.markdown("Monte Carlo simulation for Equity curve")

    col01, col02, col03, col04 = st.columns(4)
    with col01:
        start_date = st.date_input('Start Date', min_value=None, max_value=None, key=None)
    with col02:
        period_years = st.number_input('Years', min_value=1, step=1, value=1)
    with col03:
        num_trades = st.number_input('Number of Trades', min_value=10, step=10, value=150)
    with col04:
        initial_cash = st.number_input("Initial cash $", min_value=1000, step=100, value=10000)

    end_date = (start_date + timedelta(days=period_years*365)) if start_date else None
    st.text(f'End Date: {end_date.strftime("%Y-%m-%d") if end_date else ""}')

    generate_equity(start_date, end_date, num_trades, initial_cash)

    pass


if __name__ == '__main__':
    main()