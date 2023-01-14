# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:17:46 2022

@author: jnlee0417
"""

from datetime import timedelta, datetime
#from datetime import datetime
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from permetrics.regression import RegressionMetric
import os
import datetime 
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

def split(batch, n_features, n_labels):
    inputs = batch[:, :-1, :n_features]
    labels = batch[:, -1, n_features:]
    return inputs, labels

def make_dataset(data, features, labels, sequence_length, batch_size=64, shuf=False):
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data[features + labels].astype(np.float32),
        targets=None,
        sequence_length=sequence_length + 1,
        sequence_stride=1,
        shuffle=False if shuf else True,
        batch_size=batch_size,
        seed=SEED
    )
    ds = ds.map(lambda x: split(x, len(features), len(labels)))
    return ds

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def result_estimate(ob,si):
    global NSE,RMSE, R2, PBIAS
    x = ob.to_numpy()
    y = si.to_numpy()
    evaluator = RegressionMetric(x, y, decimal=4)
    NSE=evaluator.nash_sutcliffe_efficiency()
    RMSE=evaluator.root_mean_squared_error()
    R2=evaluator.pearson_correlation_coefficient()
    PBIAS=round((np.sum(y)-np.sum(x))/np.sum(x)*100,5)
    return NSE, RMSE, R2, PBIAS

def show_graph_day_lstm(yy1, yy2,sty,ety):
    fig, (ax1) = plt.subplots(1, 1,figsize=(20,4))
    plt.title("lenth: "+ str(sequence_length)+  "  unit : "+ str(unit_num)) ##
    plt.xlabel("Date(day)")
    plt.ylabel(r'Runoff ($m^3$/s)')
    ax1.semilogy(df['date'], df['obs_cms'],label='Obs value ($m^3$/s)',linewidth=0.7,color='grey')
    ax1.semilogy(df['date'], df['sim_cms'],label='Sim value ($m^3$/s)',linewidth=0.7,color='k')
    plt.text(pd.to_datetime('2005-01-01', format = '%Y-%m-%d'),pow(10,4.5), '                     NSE, RMSE, R2, PBIAS',fontsize=14) ##
    plt.text(pd.to_datetime('2005-01-01', format = '%Y-%m-%d'),pow(10,4.0), train_period,fontsize=14) ##
    plt.text(pd.to_datetime('2005-01-01', format = '%Y-%m-%d'),pow(10,3.5), valid_period,fontsize=14) ##
    plt.xlim([pd.to_datetime(sty, format = '%Y-%m-%d'),pd.to_datetime(ety, format = '%Y-%m-%d')])
    plt.legend(loc='upper left')
    plt.ylim(pow(10,-1),pow(10,5))
    plt.tight_layout()
    plt.savefig('D:/OneDrive/ONE_model/fig/'+str(NSE)+'_'+str(R2)+'_.jpg') ##
    plt.show()

SEED = 47
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

###(yd: #unit_num=75 #sequence_length =21); (sy: #unit_num=87 #sequence_length =21)
station="sy"
F_name = str(os.getcwd())+'\\datasets\\'+station+".csv"
df0 = pd.read_csv(F_name)
df0['date'] = pd.to_datetime(df0['date'])
year = 24*60*60*365.2425
timestamp = df0['date'].apply(datetime.datetime.timestamp)
df0['Year sin'] = np.sin(timestamp * (2 * np.pi / year))
df0['Year cos'] = np.cos(timestamp * (2 * np.pi / year))
df0['date'].dt.year.unique()

train_df = df0.loc[df0['date'].dt.year.isin(range(2002, 2015 + 1)), ['rf', 'inflow','etp', 'Year sin', 'Year cos']]
valid_df = df0.loc[df0['date'].dt.year.isin(range(2016, 2021 + 1)), ['rf', 'inflow', 'etp', 'Year sin', 'Year cos']]
total_df = df0.loc[df0['date'].dt.year.isin(range(2002, 2021 + 1)), ['rf', 'inflow', 'etp', 'Year sin', 'Year cos']]
train_mean = df0[['rf', 'inflow', 'etp']].mean()
train_std = df0[['rf', 'inflow', 'etp']].std()
train_df[['rf', 'inflow', 'etp']] = (train_df[['rf', 'inflow', 'etp']] - train_mean) /train_std
valid_df[['rf', 'inflow', 'etp']] = (valid_df[['rf', 'inflow', 'etp']] - train_mean) /train_std
total_df[['rf', 'inflow', 'etp']] = (total_df[['rf', 'inflow', 'etp']] - train_mean) /train_std
features = ['Year sin','Year cos','rf','etp']
labels = ['inflow']

#for k in range(500):
    #unit_num=random.randint(0,300)
    #sequence_length =random.randint(17,22)
unit_num=87
sequence_length =21
    
dataset_train = make_dataset(train_df, features, labels, sequence_length,batch_size=64)
dataset_valid = make_dataset(valid_df, features, labels, sequence_length,batch_size=64)
for batch in dataset_train.take(1):
    inputs, targets = batch
    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)

tf.random.set_seed(SEED)
inputs = Input(shape=(inputs.shape[1], inputs.shape[2]))
x = inputs
x = LSTM(unit_num)(x)
x = Dense(1)(x)
y = x
outputs = y
opt = Adam(learning_rate=0.001)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=opt, loss='mse')
model.summary()
es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=1e-4, patience=5)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./log',histogram_freq=1)
history = model.fit(dataset_train,epochs=500,validation_data=dataset_valid,callbacks=[es_callback, tensorboard_callback],)
visualize_loss(history, "Training and Validation Loss")

dataset_total = make_dataset(total_df, features, labels, sequence_length,batch_size=1, shuf=True)
predict = model.predict(dataset_total).flatten()
obs = np.concatenate([y for x, y in dataset_total]).flatten()
df = pd.DataFrame({'obs':obs, 'predict':predict})
df = df * train_std['inflow'] + train_mean['inflow']
df.rename(columns = {'obs':'obs_cms','predict':'sim_cms'},inplace=True)

##proc_sequence_length
df['date']=pd.date_range(start=datetime.datetime(2002, 1, 1) + timedelta(days=sequence_length), end= '2021-12-31',freq='D')
LSTM=df[['date','obs_cms','sim_cms']]
if station=="sy": area=2703
if station=="yd": area=930
LSTM['obs_mm'] = df['obs_cms']*86.4/area
LSTM['sim_mm'] = df['sim_cms']*86.4/area
LSTM=np.round(LSTM,3)
LSTM=LSTM.shift(periods=sequence_length)
LSTM.to_excel("D:/OneDrive/ONE_model/results/LSTM_output_"+station+".xlsx")
#print(LSTM)
#grp=LSTM.groupby('yyyy')['rf','etp','et','sim_mm','obs_mm'].sum()
#raw['yyyy'] = raw['date'].dt.year


##train
stt='2002-01-01'
ett='2015-12-31'
tmp =LSTM.loc[LSTM['date'].between(stt, ett)]
result_estimate(tmp['obs_cms'],tmp['sim_cms'])
train_period=str('train_period: ')+str('NSE:')+str(round(NSE,2))+str(' R2:')+str(round(R2,2))\
                                  +str(' RMSE:')+str(round(RMSE,2))+str(' PBIAS:')+str(round(PBIAS,2))
print(train_period)
                                  
##val
stt='2016-01-01'
ett='2021-12-31'
tmp =LSTM.loc[LSTM['date'].between(stt, ett)]
result_estimate(tmp['obs_cms'],tmp['sim_cms'])
valid_period=str('valid_period: ')+str('NSE:')+str(round(NSE,2))+str(' R2:')+str(round(R2,2))\
                                  +str(' RMSE:')+str(round(RMSE,2))+str(' PBIAS:')+str(round(PBIAS,2))
print(valid_period)

##total
stt='2002-01-01'
ett='2021-12-31'
tmp =LSTM.loc[LSTM['date'].between(stt, ett)]
result_estimate(tmp['obs_cms'],tmp['sim_cms'])
print('total_period:', 'NSE:',round(NSE,2), 'R2:',round(R2,2), 'RMSE:',round(RMSE,2), 'PBIAS:',round(PBIAS,2))

##graph
stt='2002-01-01'
ett='2021-12-31'
show_graph_day_lstm(LSTM['obs_cms'],LSTM['sim_cms'],stt,ett)


