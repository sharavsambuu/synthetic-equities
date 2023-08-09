#%%
from sys import argv
import warnings
from datetime import datetime
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import mplfinance        as mpf
warnings.filterwarnings('ignore')


#%%


#%%
asset = "btcusdt".upper()

df = pd.read_csv(f"./data/{asset}/{asset}-1m.csv", parse_dates=True, index_col="timestamp")
df = df.rename(columns={"op": "Open", "hi": "High", "lo":"Low", "cl":"Close", "volume": "Volume"})

df

#%%


#%%
df['Close'].hist(bins=150)

#%%


#%%
df['Close'].plot()


#%%


#%%
#size = 50000
#temp_df = df.iloc[-size:]
temp_df = df

#temp_df

#%%
temp_df['Close'].plot()

#%%
temp_df['Close'].hist(bins=150)

#%%


#%%
# Geometric Brownian motion generator
def gbm(mu, sigma, x0, n, dt):
    step = np.exp( (mu - sigma**2 / 2) * dt ) * np.exp( sigma * np.random.normal(0, np.sqrt(dt), (1, n)))
    return x0 * step.cumprod()


#%%
# Simple GBM exp

mu    = 1.5  # дундаж
sigma = 1.9  # хазайлт
x0    = 1.0  # эхлэх утга
n     = 100  # цувааны тоо
dt    = 1.5 # алхам

series = gbm(mu, sigma, x0, n, dt)
log_series = np.log(series)

series_df = pd.DataFrame()
series_df['val'] = log_series

series_df['val'].plot()


#%%


#%%


#%%
# Estimate mu just from the series end-points
# Note this is for a linear drift-diffusion process, i.e. the log of GBM
def simple_estimate_mu(series):
    x0 = series[0]
    T  = len(series)*dt
    return (series[-1] - x0)/T

# Use all the increments combined (maximum likelihood estimator)
# Note this is for a linear drift-diffusion process, i.e. the log of GBM
def incremental_estimate_mu(series):
    T     = len(series)*dt
    ts    = np.linspace(dt, T, len(series))
    total = (1.0 / dt) * (ts**2).sum()
    return (1.0 / total) * (1.0 / dt) * ( ts * series ).sum()

# This just estimates the sigma by its definition as the infinitesimal variance (simple Monte Carlo)
# Note this is for a linear drift-diffusion process, i.e. the log of GBM
# One can do better than this of course (MLE?)
def estimate_sigma(series):
    return np.sqrt( ( np.diff(series)**2 ).sum() / (len(series) * dt) )

# Estimator helper
all_estimates0 = lambda s: (simple_estimate_mu(s), incremental_estimate_mu(s), estimate_sigma(s))

# Since log-GBM is a linear Ito drift-diffusion process (scaled Wiener process with drift), we
# take the log of the realizations, compute mu and sigma, and then translate the mu and sigma
# to that of the GBM (instead of the log-GBM). (For sigma, nothing is required in this simple case).
def gbm_drift(log_mu, log_sigma):
    return log_mu + 0.5 * log_sigma**2

# Translates all the estimates from the log-series
def all_estimates(es):
    lmu1, lmu2, sigma = all_estimates0(es)
    return gbm_drift(lmu1, sigma), gbm_drift(lmu2, sigma), sigma


#%%


#%%
# Using one series
print("Real Mu    : {mu}")
print("Real Sigma : {sigma}")

series     = gbm(mu, sigma, x0, n, dt)
log_series = np.log(series)

print('Using 1 series: mu1 = %.2f, mu2 = %.2f, sigma = %.2f' % all_estimates(log_series) )


#%%
# Using K series
K    = 10000
s    = [ np.log(gbm(mu, sigma, x0, n, dt)) for i in range(K) ]
e    = np.array( [ all_estimates(si) for si in s ] )
avgs = np.mean(e, axis=0)

print('Using %d series: mu1 = %.2f, mu2 = %.2f, sigma = %.2f' % (K, avgs[0], avgs[1], avgs[2]) )


#%%


#%%
len(log_series)

#%%


#%%
#temp_df = df.iloc[-50000:]

n = 10000 # group size
m = 300  # overlap size

s     = [np.log(temp_df['Close'].iloc[i:i+n]) for i in range(0, len(temp_df), n-m)]
e     = np.array([all_estimates(si) for si in s])
avgs  = np.mean(e, axis=0)

print(f"mu1={avgs[0]}, mu2={avgs[1]}, sigma={avgs[2]}")


#%%


#%%


#%%
# Generate series which imitiates btcusdt behaviour

fitted_mu     = avgs[0]
fitted_sigma  = avgs[2]
x0            = 21000.0
series_len    = 1500

series_df = pd.DataFrame()
series_df['val'] = gbm(fitted_mu, fitted_sigma, x0, series_len, dt)

#series_df['val'].plot()

# OHLCV conversion

timeframe = 5
df_ = pd.DataFrame()
df_['price'] = series_df['val']

start_dt = datetime.strptime("12/1/2020 11:12:00.000000", "%d/%m/%Y %H:%M:%S.%f")
df_['datetime'] = [pd.to_datetime(start_dt+pd.DateOffset(minutes=offset)) for offset in range(0, len(df_))]

df_ = df_.set_index(pd.DatetimeIndex(df_['datetime']))
df_ = df_.drop(['datetime'], axis=1)

df_xm = df_['price'].resample(f"{timeframe}Min").ohlc()
df_xm['Volume'] = 1.0 + np.random.sample(len(df_xm)) * 15
df_xm = df_xm.rename(columns={"open": "Open", "high": "High", "low":"Low", "close":"Close"})

plot_len = 1700
mpf.plot(df_xm.iloc[-plot_len:], type='candle', style='yahoo', volume=False)


#%%


#%%


#%%
x0            = 18870    # initial price
fitted_mu     = avgs[0]
fitted_sigma  = avgs[2]
series_len    = 1500     # approx. 1 day
simulation_n  = 1000
mc_paths      = []
start_dt      = datetime.strptime("21/9/2022 08:55:00.000000", "%d/%m/%Y %H:%M:%S.%f")


for _ in range(0, simulation_n):
    series_df = pd.DataFrame()
    series_df['val'] = gbm(fitted_mu, fitted_sigma, x0, series_len, dt)
    timeframe = 5
    df_ = pd.DataFrame()
    df_['price'] = series_df['val']

    df_['datetime'] = [pd.to_datetime(start_dt+pd.DateOffset(minutes=offset)) for offset in range(0, len(df_))]

    df_ = df_.set_index(pd.DatetimeIndex(df_['datetime']))
    df_ = df_.drop(['datetime'], axis=1)

    df_xm = df_['price'].resample(f"{timeframe}Min").ohlc()
    df_xm['Volume'] = 1.0 + np.random.sample(len(df_xm)) * 15
    df_xm = df_xm.rename(columns={"open": "Open", "high": "High", "low":"Low", "close":"Close"})

    mc_paths.append(df_xm['Close'])


#%%
f, axs = plt.subplots(1, figsize=(20,15))
for series in mc_paths:
    axs.plot(series)

#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%

