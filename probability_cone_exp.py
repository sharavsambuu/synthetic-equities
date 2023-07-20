#%%
import pandas            as pd
import numpy             as np
import quantstats        as qs
import matplotlib.pyplot as plt
from   scipy.stats       import norm
from   datetime          import timedelta, datetime as dt

#%%


#%%


#%%
initial_cash = 10000.0
period_years = 3
num_trades   = 300
start_date   = dt.strptime("2018-01-12", '%Y-%m-%d').date()
end_date     = (start_date + timedelta(days=period_years*365))
trade_dates  = pd.to_datetime(np.sort(np.random.choice(pd.date_range(start=start_date, end=end_date, periods=num_trades + 1), num_trades, replace=False)))

percentage_changes = np.random.uniform(-0.043, 0.045, num_trades).astype(float)
df = pd.DataFrame({
    'datetime'  : trade_dates,
    'pct_change': percentage_changes,
})
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')
df['cumret'] = df['pct_change'].cumsum()
df['cash'  ] = (1+df['pct_change']).cumprod()*initial_cash

df['max_cash'] = df['cash'].cummax()
df['is_max'  ] = df['cash'] == df['max_cash']
higher_high_df = df[df['is_max']==True].copy()
higher_high_df['s'] = 100

df['cash'].plot()


#%%


#%%


#%%
df['cumret'].plot()


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


#%%


#%%


#%%


#%%


#%%


#%%


#%%

