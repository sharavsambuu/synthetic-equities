#%%
from   os                import path
from   datetime          import datetime
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import mplfinance        as mpf

from ydata_synthetic.synthesizers                   import ModelParameters
from ydata_synthetic.preprocessing.timeseries       import processed_stock
from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading
from ydata_synthetic.synthesizers.timeseries        import TimeGAN


#%%


#%%
#Specific to TimeGANs
seq_len       = 24
n_seq         = 5  # ohlcv
hidden_dim    = 24
gamma         = 1

noise_dim     = 32
dim           = 128
batch_size    = 128

log_step      = 100
learning_rate = 5e-4

gan_args = ModelParameters(batch_size = batch_size,
                           lr         = learning_rate,
                           noise_dim  = noise_dim,
                           layers_dim = dim)

#%%


#%%
asset = "btcusdt".upper()

df = pd.read_csv(f"./data/{asset}/{asset}-1m.csv", parse_dates=True, index_col="timestamp")
df = df.rename(columns={"op": "Open", "hi": "High", "lo":"Low", "cl":"Close", "volume": "Volume"})

df


#%%

#%%
temp_data      = df.iloc[-200:].values
temp_processed = real_data_loading(data=temp_data, seq_len=seq_len)

temp_processed

#%%


#%%
if path.exists('synthesizer_stock.pkl'):
    synth = TimeGAN.load('synthesizer_stock.pkl')
else:
    synth = TimeGAN(model_parameters=gan_args, hidden_dim=24, seq_len=seq_len, n_seq=n_seq, gamma=1)
    synth.train(temp_processed, train_steps=500)
    synth.save('synthesizer_stock.pkl')


#%%


#%%
synthetic_data = synth.sample(10)

synthetic_data


#%%
synthetic_data.shape

#%%


#%%
random_idx = np.random.randint(len(synthetic_data))
temp_df = pd.DataFrame(synthetic_data[random_idx], columns=['Open', 'High', 'Low', 'Close', 'Volume'])

start_dt = datetime.strptime("6/8/2022 15:04:00.000000", "%d/%m/%Y %H:%M:%S.%f")
temp_df['datetime'] = [pd.to_datetime(start_dt+pd.DateOffset(minutes=offset)) for offset in range(0, len(temp_df))]

temp_df = temp_df.set_index(pd.DatetimeIndex(temp_df['datetime']))
temp_df = temp_df.drop(['datetime'], axis=1)

mpf.plot(temp_df, type='candle', style='yahoo', volume=True)


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

