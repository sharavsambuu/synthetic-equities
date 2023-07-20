
import multiprocessing
import streamlit         as st
import pandas            as pd
import numpy             as np
import quantstats        as qs
import matplotlib.pyplot as plt
from   datetime          import timedelta
from   backtesting       import Backtest, Strategy
from   multiprocessing   import Pool


def mc_equity(param):
    orig_changes = param
    returns = np.zeros((orig_changes.shape[0],), dtype=float)
    for i in range(0, orig_changes.shape[0]):
        returns[i] = np.random.choice(orig_changes)
    return returns


def simulate_equity_mc(df, num_simulations, initial_cash, ruining_threshold):
    params = [df['pct_change'].values.astype(float) for _ in range(0, num_simulations)]
    result_list = []
    for param in params:
        result = mc_equity(param)
        result_list.append(result)

    sim_df = df.copy()

    df_list = []
    for idx in range(0, len(result_list)):
        temp_df = pd.DataFrame(index=df.index)
        temp_df['pct_change'] = result_list[idx]
        temp_df['cash'      ] = (1+temp_df['pct_change']).cumprod()*initial_cash
        sim_df[f"cash{idx}" ] = temp_df['cash']
        df_list.append(temp_df)

    sim_cols = [col_name for col_name in sim_df.columns if col_name.startswith("cash")]
    fig, ax = plt.subplots()
    for col_name in sim_cols:
        ax.plot(sim_df[col_name])
    st.pyplot(fig)

    stats_df = pd.DataFrame()
    stats_df['Equity'       ] = [round(df_.iloc[-1]['cash'],0) for df_ in df_list]
    stats_df['Sharpe Ratio' ] = [qs.stats.sharpe       (df_['pct_change']) for df_ in df_list]
    stats_df['Profit Factor'] = [qs.stats.profit_factor(df_['pct_change']) for df_ in df_list]
    stats_df['Max Drawdown' ] = [qs.stats.max_drawdown (df_['pct_change']) for df_ in df_list]

    ruined_simulations     = len(stats_df[stats_df['Max Drawdown']<=ruining_threshold])
    probability_of_ruining = round(ruined_simulations/num_simulations*100.0, 0)
    st.text(f"Probability of ruining : {probability_of_ruining}%")

    st.dataframe(stats_df)

    pass


def generate_equity(start_date, end_date, num_trades, initial_cash):
    trade_dates = pd.to_datetime(np.sort(np.random.choice(pd.date_range(start=start_date, end=end_date, periods=num_trades + 1), num_trades, replace=False)))
    percentage_changes = np.random.uniform(-0.043, 0.045, num_trades).astype(float)
    df = pd.DataFrame({
        'datetime'  : trade_dates,
        'pct_change': percentage_changes,
    })
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df['cumret' ] = df['pct_change'].cumsum()
    df['cash'] = (1+df['pct_change']).cumprod()*initial_cash

    df['max_cash'] = df['cash'].cummax()
    df['is_max'  ] = df['cash'] == df['max_cash']
    higher_high_df = df[df['is_max']==True].copy()
    higher_high_df['s'] = 100

    fig, ax = plt.subplots()
    ax.plot(df['cash'])
    ax.scatter(higher_high_df.index, higher_high_df['cash'], color='green', s=higher_high_df['s'])
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

    col11, col12, _ = st.columns(3)
    with col11:
        num_simulations = st.number_input("N Simulations", min_value=10, step=5, value=90)
    with col12:
        ruining_threshold = st.number_input("Ruin drawdown threshold", min_value=-1.0, step=0.01, value=-0.3)

    simulate_equity_mc(df, num_simulations, initial_cash, ruining_threshold)



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