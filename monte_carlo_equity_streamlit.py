import multiprocessing
import streamlit         as st
import pandas            as pd
import numpy             as np
import quantstats        as qs
import matplotlib.pyplot as plt
from   datetime          import timedelta
from   backtesting       import Backtest, Strategy
from   multiprocessing   import Pool


def generate_equity(start_date, end_date, num_trades):
    trade_dates = pd.to_datetime(np.sort(np.random.choice(pd.date_range(start=start_date, end=end_date, periods=num_trades + 1), num_trades, replace=False)))
    percentage_changes = np.random.uniform(-0.043, 0.045, num_trades)
    df = pd.DataFrame({
        'datetime'  : trade_dates,
        'pct_change': percentage_changes,
    })
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df['cumret' ] = df['pct_change'].cumsum()

    fig, ax = plt.subplots()
    ax.plot(df['cumret'])
    st.pyplot(fig)

    sharpe_ratio  = qs.stats.sharpe(df['pct_change'])
    profit_factor = qs.stats.profit_factor(df['pct_change'])
    max_drawdown  = qs.stats.max_drawdown(df['pct_change'])

    stats_df = pd.DataFrame()
    stats_df['Sharpe Ratio' ] = [sharpe_ratio ]
    stats_df['Profit Factor'] = [profit_factor]
    stats_df['Max Drawdown' ] = [max_drawdown ]
    st.dataframe(stats_df)




def main():
    st.markdown("Monte Carlo simulation for Equity curve")

    col01, col02, col03 = st.columns(3)
    with col01:
        start_date = st.date_input('Start Date', min_value=None, max_value=None, key=None)
    with col02:
        period_years = st.number_input('Years', min_value=1, step=1, value=1)
    with col03:
        num_trades = st.number_input('Number of Trades', min_value=10, step=10, value=150)
        pass

    end_date = (start_date + timedelta(days=period_years*365)) if start_date else None
    st.text(f'End Date: {end_date.strftime("%Y-%m-%d") if end_date else ""}')

    generate_equity(start_date, end_date, num_trades)

    pass


if __name__ == '__main__':
    main()