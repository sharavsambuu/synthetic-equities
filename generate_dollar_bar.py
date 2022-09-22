#%%
import sys
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy  as np


#%%
asset         = str(sys.argv[1]).upper()
dollar_amount = float(sys.argv[2])

#%%
df = pd.read_csv(f"./data/{asset}/{asset}-1m.csv", parse_dates=True, index_col="timestamp")
df = df.dropna()
df = df.rename(columns={"op":"Open", "hi":"High", "lo":"Low", "cl":"Close", "volume":"Volume"})

df = df["2018-01-12":]

print(df)

#%%


#%%
class BarSeries(object):
    def __init__(self, df, datetimecolumn='timestamp'):
        self.df             = df
        self.datetimecolumn = datetimecolumn
    def process_column(self, column_name, frequency):
        return self.df[column_name].resample(frequency, label='right').ohlc()
    def process_volume(self, column_name, frequency):
        return self.df[column_name].resample(frequency, label='right').sum()
    def process_ticks(self, price_column='price', volume_column='size', frequency='5Min'):
        price_df  = self.process_column(price_column, frequency)
        volume_df = self.process_volume(volume_column, frequency)
        price_df['volume'] = volume_df
        return price_df

class DollarBarSeries(BarSeries):
    def __init__(self, df, datetimecolumn='timestamp', volume_column='Volume'):
        super(DollarBarSeries, self).__init__(df, datetimecolumn)
        self.volume_column = volume_column
    def process_column(self, column_name, frequency):
        res         = []
        buf, vbuf   = [], []
        start_index = 0.
        dollar_buf  = 0.
        for i in range(len(self.df[column_name])):
            di  = self.df.index.values[i]
            pi  = self.df[column_name].iloc[i]
            vi  = self.df[self.volume_column].iloc[i] 
            dvi = pi*vi
            buf.append(pi)
            vbuf.append(vi)
            dollar_buf += dvi
            if dollar_buf>=frequency:
                o = buf[0]
                h = np.max(buf)
                l = np.min(buf)
                c = buf[-1]
                v = np.sum(vbuf)
                res.append({
                    self.datetimecolumn: di,
                    'Open'   : o,
                    'High'   : h,
                    'Low'    : l,
                    'Close'  : c,
                    'Volume' : v,
                    'Dollar' : dollar_buf
                })
                buf, vbuf, dollar_buf = [], [], 0
        res = pd.DataFrame(res)
        #res = res.dropna()
        res = res.set_index(self.datetimecolumn)
        return res
    def process_ticks(self, price_column='Close', volume_column='Volume', frequency=100000):
        price_df = self.process_column(price_column, frequency)
        return price_df


#%%


#%%

df['timestamp'] = df.index
#df['HL2'      ] = (df['High']+df['Low'])/2.0

dollarbar_series = DollarBarSeries(df)

dollar_df     = dollarbar_series.process_ticks(
    price_column  = 'Close' , #'HL2'
    volume_column = 'Volume', 
    frequency     = dollar_amount
    ) 


#%%
dollar_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Dollar']].to_csv(f"./data/{asset}/{asset}-dollarbar-{int(dollar_amount)}.csv", index=True, header=True)

#%%


#%%
loaded_df = pd.read_csv(f"./data/{asset}/{asset}-dollarbar-{int(dollar_amount)}.csv", parse_dates=True, index_col="timestamp")

print(loaded_df)

print('\n')

print(loaded_df['Dollar'].mean())


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
