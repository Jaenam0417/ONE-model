# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 13:21:14 2022

@author: jnlee0417
"""

import os
import random
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from permetrics.regression import RegressionMetric
#https://github.com/thieu1995/permetrics



def run_setting(station):
    global area,F_name, last_day_of_the_month
    last_day_of_the_month = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    if station=="sy":
        area=2703
    if station=="yd":
        area=930
    return area, F_name, last_day_of_the_month

def data_read(syear, eyear):
    global rf,ep, sw, et, qq, obs, df, F_name
    #raw = pd.read_excel(F_name+".xlsx",sheet_name="input")
    #raw = pd.read_csv('"'+ D:\\OneDrive\\ONE_model\\datasets\\yd.csv')
    F_name = str(os.getcwd())+'\\datasets\\'+station+".csv"
    print(F_name)
    
    raw = pd.read_csv(F_name)
    print(F_name)
    raw["nan"]=np.nan
    raw['date'] = pd.to_datetime(raw['date'].copy())
    raw['yyyy'] = raw['date'].dt.year
    ##resetting
    df=raw[(raw['yyyy'] >= syear)&(raw['yyyy'] <= eyear)].reset_index(drop=True)
    rf=df[['rf']].copy()
    ep=df[['etp']].copy()
    sw=df[["nan"]].copy()
    et=df[["nan"]].copy()
    qq=df[["nan"]].copy()
    obs=df[["inflow"]].copy()

def calcaul(start_year, end_year, para_w):
    k = 0
    sw.iloc[0,0] = 70
    for yyyy in range(start_year, end_year + 1):
        dayList = np.array(last_day_of_the_month)
        if calendar.isleap(yyyy) == True:
            dayList[1] = 29    
        elif calendar.isleap(yyyy) == False:  
            dayList[1] = 28
        
        for mm in range(1,13):
            for dd in range(1, dayList[mm-1] + 1):
                if k == 0:
                    sw.iloc[k,0] = sw.iloc[k,0] + rf.iloc[k,0]
                else:
                    sw.iloc[k,0] = sw.iloc[k-1,0] + rf.iloc[k,0]
                
                et.iloc[k,0] = ep.iloc[k,0] * (1-np.exp(-0.015*sw.iloc[k,0]))
                qq.iloc[k,0] = sw.iloc[k,0]*pow(1 - np.exp(-0.003* sw.iloc[k,0]), (0.2 + np.exp(-0.001 * sw.iloc[k,0]) * para_w))
                if sw.iloc[k,0] > 30:
                    sw.iloc[k,0] = sw.iloc[k,0] - et.iloc[k,0] - qq.iloc[k,0]
                else:
                    sw.iloc[k,0] = sw.iloc[k,0] - qq.iloc[k,0]            
                k = k + 1

def simulation_result():
    global rf,ep, sw, et, qq, obs, df, area
    df['rf'] = rf.iloc[:,0]
    df['sw'] = sw.iloc[:,0]
    df['et'] = et.iloc[:,0]
    df['qq'] = qq.iloc[:,0] 
    
    df['sim_mm'] = df['qq']
    df['sim_cms'] = df['sim_mm']*area/86.4
    df['obs_cms'] = df['inflow']
    df['obs_mm'] = df['obs_cms']*86.4/area
    
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
    
def show_graph_line(xx, yy):
    fig, (ax1) = plt.subplots(1, 1,figsize=(4.5,3.5))
    plt.rcdefaults()
    rg=max(max(xx),max(yy))*1.3
    x=np.arange(0,rg); y=x
    ax1.plot(x,y,color="k",alpha=0.5,linewidth=1,linestyle='solid')
    plt.xlabel('Obs.value (mm)', fontsize=10)
    plt.ylabel('Sim.value (mm)', fontsize=10)
    im=ax1.scatter(xx,yy,c=yy,alpha = 0.5, s = 11, edgecolors="k", cmap = 'Spectral')
    plt.ylim(0,rg)
    plt.xlim(0,rg)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.colorbar(im)
    fig.tight_layout()
    plt.show()

def show_graph_day(yy1, yy2, sty, ety):
    fig, (ax1) = plt.subplots(1, 1,figsize=(20,4))
    plt.xlabel("Date(day)")
    plt.ylabel(r'Runoff ($m^3$/s)')
    ax1.semilogy(df['date'], yy1,label='Obs value ($m^3$/s)',linewidth=1.5,color='grey',alpha = 1)
    ax1.semilogy(df['date'], yy2,label='Sim value ($m^3$/s)',linewidth=1.1,color='darkred',alpha = 1)
    plt.xlim([pd.to_datetime(sty, format = '%Y-%m-%d'),pd.to_datetime(ety, format = '%Y-%m-%d')])
    plt.legend(loc='upper left')
    plt.ylim(pow(10,-1),pow(10,5))
    plt.tight_layout()
    plt.show()

###ONE_model##########################################################################################
##yd=3.5>>2.96  ##sy=2.5
station='yd'
para_w=2.96

syear=2002
eyear=2021
run_setting(station)
data_read(syear, eyear)
calcaul(syear, eyear, para_w)
simulation_result()
###ONE_model##########################################################################################

result_estimate(df['obs_mm'],df['sim_mm'])
print(str(NSE),str(R2),str(RMSE),str(PBIAS))
grp=df.groupby('yyyy')['rf','etp','et','sim_mm','obs_mm'].sum()
print(np.round(grp,2))
ONE=df[['date','rf','sw','sim_mm','sim_cms','obs_mm','obs_cms']]
ONE=np.round(ONE,3)
ONE.to_excel('D:/OneDrive/ONE_model/results/ONE_output_'+station+'.xlsx')

##cal
stt='2002-01-01'
ett='2015-12-31'
tmp =df.loc[df['date'].between(stt, ett)]
show_graph_line(tmp['obs_cms'],tmp['sim_cms'])
result_estimate(tmp['obs_cms'],tmp['sim_cms'])
print('calib_period:', 'NSE:',round(NSE,2), 'R2:',round(R2,2), 'RMSE:',round(RMSE,2), 'PBIAS:',round(PBIAS,2))
show_graph_day(ONE['obs_cms'],df['sim_cms'],stt,ett)

##val
stt='2016-01-01'
ett='2021-12-31'
show_graph_day(ONE['obs_cms'],ONE['sim_cms'],stt,ett)
tmp =df.loc[df['date'].between(stt, ett)]
result_estimate(tmp['obs_cms'],tmp['sim_cms'])
print('valid_period:', 'NSE:',round(NSE,2), 'R2:',round(R2,2), 'RMSE:',round(RMSE,2), 'PBIAS:',round(PBIAS,2))
show_graph_line(tmp['obs_cms'],tmp['sim_cms'])

##total
stt='2002-01-01'
ett='2021-12-31'
tmp =df.loc[ONE['date'].between(stt, ett)]
show_graph_line(tmp['obs_cms'],tmp['sim_cms'])
result_estimate(tmp['obs_cms'],tmp['sim_cms'])
print('total_period:', 'NSE:',round(NSE,2), 'R2:',round(R2,2), 'RMSE:',round(RMSE,2), 'PBIAS:',round(PBIAS,2))
show_graph_day(ONE['obs_cms'],ONE['sim_cms'],stt,ett)

