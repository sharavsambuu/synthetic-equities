
import multiprocessing
import streamlit         as st
import pandas            as pd
import numpy             as np
import quantstats        as qs
import matplotlib.pyplot as plt
from   scipy             import stats as scipy_stats
from   datetime          import timedelta
from   backtesting       import Backtest, Strategy
from   multiprocessing   import Pool



# functions for SR, PSR, DSR
#


def estimated_sharpe_ratio(returns):
    """
    Calculate the estimated sharpe ratio (risk_free=0).
    Parameters
    ----------
    returns: np.array, pd.Series, pd.DataFrame
    Returns
    -------
    float, pd.Series
    """
    return returns.mean() / returns.std(ddof=1)


def ann_estimated_sharpe_ratio(returns=None, periods=261, *, sr=None):
    """
    Calculate the annualized estimated sharpe ratio (risk_free=0).
    Parameters
    ----------
    returns: np.array, pd.Series, pd.DataFrame
    periods: int
        How many items in `returns` complete a Year.
        If returns are daily: 261, weekly: 52, monthly: 12, ...
    sr: float, np.array, pd.Series, pd.DataFrame
        Sharpe ratio to be annualized, it's frequency must be coherent with `periods`
    Returns
    -------
    float, pd.Series
    """
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    sr = sr * np.sqrt(periods)
    return sr


def estimated_sharpe_ratio_stdev(returns=None, *, n=None, skew=None, kurtosis=None, sr=None):
    """
    Calculate the standard deviation of the sharpe ratio estimation.
    Parameters
    ----------
    returns: np.array, pd.Series, pd.DataFrame
        If no `returns` are passed it is mandatory to pass the other 4 parameters.
    n: int
        Number of returns samples used for calculating `skew`, `kurtosis` and `sr`.
    skew: float, np.array, pd.Series, pd.DataFrame
        The third moment expressed in the same frequency as the other parameters.
        `skew`=0 for normal returns.
    kurtosis: float, np.array, pd.Series, pd.DataFrame
        The fourth moment expressed in the same frequency as the other parameters.
        `kurtosis`=3 for normal returns.
    sr: float, np.array, pd.Series, pd.DataFrame
        Sharpe ratio expressed in the same frequency as the other parameters.
    Returns
    -------
    float, pd.Series
    Notes
    -----
    This formula generalizes for both normal and non-normal returns.
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
    """
    if type(returns) != pd.DataFrame:
        _returns = pd.DataFrame(returns)
    else:
        _returns = returns.copy()

    if n is None:
        n = len(_returns)
    if skew is None:
        skew = pd.Series(scipy_stats.skew(_returns), index=_returns.columns)
    if kurtosis is None:
        kurtosis = pd.Series(scipy_stats.kurtosis(_returns, fisher=False), index=_returns.columns)
    if sr is None:
        sr = estimated_sharpe_ratio(_returns)

    sr_std = np.sqrt((1 + (0.5 * sr ** 2) - (skew * sr) + (((kurtosis - 3) / 4) * sr ** 2)) / (n - 1))

    if type(returns) == pd.DataFrame:
        sr_std = pd.Series(sr_std, index=returns.columns)
    elif type(sr_std) not in (float, np.float64, pd.DataFrame):
        sr_std = sr_std.values[0]

    return sr_std


def probabilistic_sharpe_ratio(returns=None, sr_benchmark=0.0, *, sr=None, sr_std=None):
    """
    Calculate the Probabilistic Sharpe Ratio (PSR).
    Parameters
    ----------
    returns: np.array, pd.Series, pd.DataFrame
        If no `returns` are passed it is mandatory to pass a `sr` and `sr_std`.
    sr_benchmark: float
        Benchmark sharpe ratio expressed in the same frequency as the other parameters.
        By default set to zero (comparing against no investment skill).
    sr: float, np.array, pd.Series, pd.DataFrame
        Sharpe ratio expressed in the same frequency as the other parameters.
    sr_std: float, np.array, pd.Series, pd.DataFrame
        Standard deviation fo the Estimated sharpe ratio,
        expressed in the same frequency as the other parameters.
    Returns
    -------
    float, pd.Series
    Notes
    -----
    PSR(SR*) = probability that SR^ > SR*
    SR^ = sharpe ratio estimated with `returns`, or `sr`
    SR* = `sr_benchmark`
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
    """
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    if sr_std is None:
        sr_std = estimated_sharpe_ratio_stdev(returns, sr=sr)

    psr = scipy_stats.norm.cdf((sr - sr_benchmark) / sr_std)

    if type(returns) == pd.DataFrame:
        psr = pd.Series(psr, index=returns.columns)
    elif type(psr) not in (float, np.float64):
        psr = psr[0]

    return psr


def min_track_record_length(returns=None, sr_benchmark=0.0, prob=0.95, *, n=None, sr=None, sr_std=None):
    """
    Calculate the MIn Track Record Length (minTRL).
    Parameters
    ----------
    returns: np.array, pd.Series, pd.DataFrame
        If no `returns` are passed it is mandatory to pass a `sr` and `sr_std`.
    sr_benchmark: float
        Benchmark sharpe ratio expressed in the same frequency as the other parameters.
        By default set to zero (comparing against no investment skill).
    prob: float
        Confidence level used for calculating the minTRL.
        Between 0 and 1, by default=0.95
    n: int
        Number of returns samples used for calculating `sr` and `sr_std`.
    sr: float, np.array, pd.Series, pd.DataFrame
        Sharpe ratio expressed in the same frequency as the other parameters.
    sr_std: float, np.array, pd.Series, pd.DataFrame
        Standard deviation fo the Estimated sharpe ratio,
        expressed in the same frequency as the other parameters.
    Returns
    -------
    float, pd.Series
    Notes
    -----
    minTRL = minimum of returns/samples needed (with same SR and SR_STD) to accomplish a PSR(SR*) > `prob`
    PSR(SR*) = probability that SR^ > SR*
    SR^ = sharpe ratio estimated with `returns`, or `sr`
    SR* = `sr_benchmark`
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
    """
    if n is None:
        n = len(returns)
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    if sr_std is None:
        sr_std = estimated_sharpe_ratio_stdev(returns, sr=sr)

    min_trl = 1 + (sr_std ** 2 * (n - 1)) * (scipy_stats.norm.ppf(prob) / (sr - sr_benchmark)) ** 2

    if type(returns) == pd.DataFrame:
        min_trl = pd.Series(min_trl, index=returns.columns)
    elif type(min_trl) not in (float, np.float64):
        min_trl = min_trl[0]

    return min_trl


def num_independent_trials(trials_returns=None, *, m=None, p=None):
    """
    Calculate the number of independent trials.
    
    Parameters
    ----------
    trials_returns: pd.DataFrame
        All trials returns, not only the independent trials.
        
    m: int
        Number of total trials.
        
    p: float
        Average correlation between all the trials.
    Returns
    -------
    int
    """
    if m is None:
        m = trials_returns.shape[1]
        
    if p is None:
        corr_matrix = trials_returns.corr()
        p = corr_matrix.values[np.triu_indices_from(corr_matrix.values,1)].mean()
        
    n = p + (1 - p) * m
    
    n = int(n)+1  # round up
    
    return n


def expected_maximum_sr(trials_returns=None, expected_mean_sr=0.0, *, independent_trials=None, trials_sr_std=None):
    """
    Compute the expected maximum Sharpe ratio (Analytically)
    
    Parameters
    ----------
    trials_returns: pd.DataFrame
        All trials returns, not only the independent trials.
        
    expected_mean_sr: float
        Expected mean SR, usually 0. We assume that random startegies will have a mean SR of 0,
        expressed in the same frequency as the other parameters.
        
    independent_trials: int
        Number of independent trials, must be between 1 and `trials_returns.shape[1]`
        
    trials_sr_std: float
        Standard deviation fo the Estimated sharpe ratios of all trials,
        expressed in the same frequency as the other parameters.
    Returns
    -------
    float
    """
    emc = 0.5772156649 # Euler-Mascheroni constant
    
    if independent_trials is None:
        independent_trials = num_independent_trials(trials_returns)
    
    if trials_sr_std is None:
        srs = estimated_sharpe_ratio(trials_returns)
        trials_sr_std = srs.std()
    
    maxZ = (1 - emc) * scipy_stats.norm.ppf(1 - 1./independent_trials) + emc * scipy_stats.norm.ppf(1 - 1./(independent_trials * np.e))
    expected_max_sr = expected_mean_sr + (trials_sr_std * maxZ)
    
    return expected_max_sr


def deflated_sharpe_ratio(trials_returns=None, returns_selected=None, expected_mean_sr=0.0, *, expected_max_sr=None):
    """
    Calculate the Deflated Sharpe Ratio (PSR).
    Parameters
    ----------
    trials_returns: pd.DataFrame
        All trials returns, not only the independent trials.
        
    returns_selected: pd.Series
    expected_mean_sr: float
        Expected mean SR, usually 0. We assume that random startegies will have a mean SR of 0,
        expressed in the same frequency as the other parameters.
        
    expected_max_sr: float
        The expected maximum sharpe ratio expected after running all the trials,
        expressed in the same frequency as the other parameters.
    Returns
    -------
    float
    Notes
    -----
    DFS = PSR(SR⁰) = probability that SR^ > SR⁰
    SR^ = sharpe ratio estimated with `returns`, or `sr`
    SR⁰ = `max_expected_sr`
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
    """
    if expected_max_sr is None:
        expected_max_sr = expected_maximum_sr(trials_returns, expected_mean_sr)
        
    dsr = probabilistic_sharpe_ratio(returns=returns_selected, sr_benchmark=expected_max_sr)

    return dsr


def moments(returns):
    """
    Calculate the four moments: mean, std, skew, kurtosis.
    Parameters
    ----------
    returns: np.array, pd.Series, pd.DataFrame
    Returns
    -------
    pd.Series, pd.DataFrame
    """
    if type(returns) != pd.DataFrame:
        return pd.Series({'mean': np.mean(returns),
                          'std': np.std(returns, ddof=1),
                          'skew': scipy_stats.skew(returns),
                          'kurt': scipy_stats.kurtosis(returns, fisher=False)})
    else:
        return returns.apply(moments, axis=1)




def mc_equity(param):
    orig_changes = param
    returns = np.zeros((orig_changes.shape[0],), dtype=float)
    for i in range(0, orig_changes.shape[0]):
        returns[i] = np.random.choice(orig_changes)
    return returns


def simulate_equity_mc(df, num_simulations, initial_cash, ruining_threshold, end_date, num_trades):
    params = [df['pct_change'].values.astype(float) for _ in range(0, num_simulations)]
    mc_result_list = []
    for param in params:
        result = mc_equity(param)
        mc_result_list.append(result)

    sim_df = df.copy()

    df_list = []
    for idx in range(0, len(mc_result_list)):
        temp_df = pd.DataFrame(index=df.index)
        temp_df['pct_change'] = mc_result_list[idx]
        temp_df['cash'      ] = (1+temp_df['pct_change']).cumprod()*initial_cash
        sim_df[f"cash{idx}" ] = temp_df['cash']
        sim_df[f"path{idx}" ] = temp_df['pct_change']
        df_list.append(temp_df)

    sim_cols  = [col_name for col_name in sim_df.columns if col_name.startswith("cash")]
    fig, ax = plt.subplots()
    for col_name in sim_cols:
        ax.plot(sim_df[col_name])
    fig.autofmt_xdate()
    st.pyplot(fig)

    # About to calculate PSR and DSR
    path_cols        = [col_name for col_name in sim_df.columns if col_name.startswith("path")]
    returns_df       = sim_df[path_cols].copy()
    returns_df.reset_index(drop=True, inplace=True)
    best_psr_name    = probabilistic_sharpe_ratio(returns=returns_df, sr_benchmark=0).sort_values(ascending=False).index[0]
    best_psr_returns = returns_df[best_psr_name]
    dsr              = deflated_sharpe_ratio(trials_returns=returns_df, returns_selected=best_psr_returns)

    stats_df = pd.DataFrame()
    stats_df['Equity'                   ] = [round(df_.iloc[-1]['cash'],0) for df_ in df_list]
    stats_df['Sharpe Ratio'             ] = [qs.stats.sharpe       (df_['pct_change']) for df_ in df_list]
    stats_df['Probablistic Sharpe Ratio'] = [probabilistic_sharpe_ratio(returns=df_['pct_change'], sr_benchmark=0.0) for df_ in df_list]
    stats_df['Profit Factor'            ] = [qs.stats.profit_factor(df_['pct_change']) for df_ in df_list]
    stats_df['Max Drawdown'             ] = [qs.stats.max_drawdown (df_['pct_change']) for df_ in df_list]

    max_drawdown           = round(stats_df['Max Drawdown' ].min()*100, 1)
    mean_drawdown          = round(stats_df['Max Drawdown' ].mean()*100, 1)
    median_drawdown        = round(stats_df['Max Drawdown' ].median()*100, 1)
    mean_sharpe_ratio      = round(stats_df['Sharpe Ratio' ].mean(), 0)
    mean_equity            = round(stats_df['Equity'       ].mean(), 0)
    mean_profit_factor     = round(stats_df['Profit Factor'].mean(), 1)
    ruined_simulations     = len(stats_df[stats_df['Max Drawdown']<=ruining_threshold])
    probability_of_ruining = round(ruined_simulations/num_simulations*100.0, 0)

    is_prob_ruining_ok = ":green" if probability_of_ruining<=1.0 else ":red"
    is_dsr_ok          = ":green" if dsr>=0.95 else ":red"

    st.markdown(f"###### If strategy is good, Probability of Ruinning and Deflated Sharpe Ratio are both should be green.")

    st.markdown(f"{is_prob_ruining_ok}[Probability of ruining &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; : {probability_of_ruining}% ]")
    st.markdown(f"{is_dsr_ok         }[Deflated Sharpe Ratio  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; : {round(dsr,2)}]")
    st.text(f"Mean Drawdown        : {mean_drawdown}%")
    st.text(f"Median Drawdown      : {median_drawdown}%")
    st.text(f"Maximum Drawdown     : {max_drawdown}%")
    st.text(f"Mean Sharpe Ratio    : {mean_sharpe_ratio}")
    st.text(f"Mean Equity          : {mean_equity}$")
    st.text(f"Mean Profit Factor   : {mean_profit_factor}")

    st.markdown("##### Simulation metrics ")
    st.dataframe(stats_df)

    #st.markdown("##### Simulation moments")
    #st.dataframe(moments(returns_df))


    # Trying to render confidence intervals with std1 and std2
    st.markdown("##### Confidence interval with 1 std and 2 std")

    equity_df = pd.DataFrame()
    equity_df['cash'] = (1+df['pct_change']).cumprod()*initial_cash
    cash_diff = equity_df.iloc[-1]['cash']-equity_df.iloc[0]['cash']

    sim_cash_df = sim_df[sim_cols].copy()
    sim_cash_df.index = pd.date_range(equity_df.index[-1], periods=len(sim_cash_df), freq='D')
    for col_name in sim_cols:
        sim_cash_df[col_name] += cash_diff

    fig, ax = plt.subplots()
    ax.plot(equity_df['cash'], color='b', label='Original Equity')
    for col_name in sim_cols:
        ax.plot(sim_cash_df[col_name], color='gray', alpha=0.1)
    fig.autofmt_xdate()
    st.pyplot(fig)

    pass


def generate_equity(start_date, end_date, num_trades, initial_cash):
    trade_dates = pd.to_datetime(np.sort(np.random.choice(pd.date_range(start=start_date, end=end_date, periods=num_trades + 1), num_trades, replace=False)))
    percentage_changes = np.random.uniform(-0.032, 0.034, num_trades).astype(float)
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
    fig.autofmt_xdate()
    st.pyplot(fig)

    sharpe_ratio  = qs.stats.sharpe       (df['pct_change'])
    prob_sr       = probabilistic_sharpe_ratio(returns=df['pct_change'], sr_benchmark=0.0)
    profit_factor = qs.stats.profit_factor(df['pct_change'])
    max_drawdown  = qs.stats.max_drawdown (df['pct_change'])

    stats_df = pd.DataFrame()
    stats_df['Equity'                    ] = [round(df.iloc[-1]['cash'],0)]
    stats_df['Sharpe Ratio'              ] = [sharpe_ratio ]
    stats_df['Probabilistic Sharpe Ratio'] = [prob_sr]
    stats_df['Profit Factor'             ] = [profit_factor]
    stats_df['Max Drawdown'              ] = [max_drawdown ]
    st.dataframe(stats_df)

    col11, col12, _ = st.columns(3)
    with col11:
        num_simulations = st.number_input("N Simulations", min_value=10, step=5, value=90)
    with col12:
        ruining_threshold = st.number_input("Ruin drawdown threshold", min_value=-1.0, step=0.01, value=-0.35)

    simulate_equity_mc(df, num_simulations, initial_cash, ruining_threshold, end_date, num_trades)


def main():
    st.markdown("### Monte Carlo Simulation of Equity Curve")

    col01, col02, col03 = st.columns(3)
    with col01:
        start_date = st.date_input('Start Date', min_value=None, max_value=None, key=None)
    with col02:
        num_trades = st.number_input('Number of Trades', min_value=10, step=10, value=300)
    with col03:
        initial_cash = st.number_input("Initial cash $", min_value=1000, step=100, value=10000)

    end_date = (start_date + timedelta(days=num_trades)) if start_date else None
    st.text(f'End Date: {end_date.strftime("%Y-%m-%d") if end_date else ""}')

    generate_equity(start_date, end_date, num_trades, initial_cash)

    pass


if __name__ == '__main__':
    main()