#%%
import random
from   os                import path
from   datetime          import datetime
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import mplfinance        as mpf

from sklearn.preprocessing                    import MinMaxScaler
from ydata_synthetic.synthesizers             import ModelParameters
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries  import TimeGAN

#np.random.seed(0)

#%%


#%%
asset = "btcusdt".upper()

df = pd.read_csv(f"./data/{asset}/{asset}-1m.csv", parse_dates=True, index_col="timestamp")
df = df.rename(columns={"op": "Open", "hi": "High", "lo":"Low", "cl":"Close", "volume": "Volume"})

df


#%%


#%%
def real_data_loading(data: np.array, seq_len):
    ori_data = data[::-1]
    scaler   = MinMaxScaler().fit(ori_data)
    ori_data = scaler.transform(ori_data)

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return data, scaler


#%%
seq_len = 300

temp_processed, data_scaler = real_data_loading(data=df.values, seq_len=seq_len)

#%%
data_scaler

#%%
temp_processed[0].shape

#%%
processed_size = len(temp_processed)

processed_size

#%%
train_samples_size = 100000
random_numbers     = random.sample(range(0, processed_size), train_samples_size)
selected_indexes   = list(dict.fromkeys(random_numbers))

len(selected_indexes)

#%%
print(list(selected_indexes[:10]))

#%%
from tqdm import tqdm

#%%
downsampled_dataset = []

for idx in tqdm(selected_indexes):
    downsampled_dataset.append(temp_processed[idx])

#%%
len(downsampled_dataset), downsampled_dataset[0].shape

#%%


#%%


#%%
#Specific to TimeGANs
n_feature     = 5  # ohlcv
hidden_dim    = 24
gamma         = 1

noise_dim     = 32
dim           = 128
batch_size    = 256 #128

log_step      = 100
learning_rate = 5e-4

gan_args = ModelParameters(batch_size = batch_size,
                           lr         = learning_rate,
                           noise_dim  = noise_dim,
                           layers_dim = dim)

#%%


#%%
if path.exists('synthesizer_stock.pkl'):
    synth = TimeGAN.load('synthesizer_stock.pkl')
else:
    synth = TimeGAN(model_parameters=gan_args, hidden_dim=24, seq_len=seq_len, n_seq=n_feature, gamma=1)
    synth.train(downsampled_dataset, train_steps=10000)
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

