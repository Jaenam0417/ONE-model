# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 02:12:18 2022

@author: melod
"""

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def show_graph_day_one_lstm(raw, stt, ett):
    data=raw
    data=data.loc[data["date"].between(stt, ett)]
    
    fig, (ax1) = plt.subplots(1, 1,figsize=(20,4))
    plt.xlabel("Date(day)")
    plt.ylabel(r'Runoff ($m^3$/s)')
    ax1.semilogy(data['date'], data['obs_cms'],label='Obs value ($m^3$/s)',linewidth=1.5,color='grey',alpha = 1)
    ax1.semilogy(data['date'], data['sim_cms_one'],label='ONE model ($m^3$/s)',linewidth=1.1,color='darkred',alpha =1)
    ax1.semilogy(data['date'], data['sim_cms_lstm'],label='LSTM model ($m^3$/s)',linewidth=0.8,color='k',alpha =0.8)
    plt.xlim([pd.to_datetime(stt, format = '%Y-%m-%d'),pd.to_datetime(ett, format = '%Y-%m-%d')])
    plt.legend(loc='upper left')
    plt.ylim(pow(10,-1),pow(10,5))
    plt.tight_layout()
    plt.show()

def show_graph_line_one_lstm(raw, stt, ett):
    data=raw
    data=data.loc[data["date"].between(stt, ett)]
    
    fig, (ax1) = plt.subplots(1, 1,figsize=(5.5,4.8))
    plt.rcdefaults()
    rg=max(max(data['obs_cms']),max(data['sim_cms_one']),max(data['sim_cms_lstm']))*1.3
    x=np.arange(0,rg); y=x
    ax1.plot(x,y,color="k",alpha=0.5,linewidth=1,linestyle='solid')
    plt.xlabel(r'Obs ($m^3$/s)', fontsize=10)
    plt.ylabel(r'Sim ($m^3$/s)', fontsize=10)
    #im1=ax1.scatter(xx,yy,c=yy, alpha = 0.5, s = 20, edgecolors="k",label='ONE model',cmap='Spectral',marker='o')
    #ax1.scatter(xx,zz,c=zz, alpha = 0.5, s = 20, edgecolors="k",label='LSTM model',cmap='Spectral',marker='^')
    ax1.scatter(data['obs_cms'],data['sim_cms_one'],c='k', alpha = 0.5, s = 20, edgecolors="k",label='ONE model')
    ax1.scatter(data['obs_cms'],data['sim_cms_lstm'],c='r', alpha = 0.5, s = 20, edgecolors="k",label='LSTM model')
    ax1.legend()
    plt.ylim(0,rg)
    plt.xlim(0,rg)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    #plt.colorbar(im1)
    fig.tight_layout()
    plt.show()

def show_graph_heatmap_one_lstm(raw):
    data=raw
    data=data[['date','obs_mm','sim_mm_one','sim_mm_lstm' ]]
    
    fig, axs = plt.subplots(1, 3, figsize=(11,5))
    data['mm']=data['date'].dt.month
    data['yy']=data['date'].dt.year
    data=data[['yy', 'mm', 'obs_mm','sim_mm_one','sim_mm_lstm']]
    heatmap_obs = pd.pivot_table(data,values="obs_mm",index=["yy"],columns="mm",aggfunc=np.sum)
    heatmap_one = pd.pivot_table(data,values="sim_mm_one",index=["yy"],columns="mm",aggfunc=np.sum)
    heatmap_lstm = pd.pivot_table(data,values="sim_mm_lstm",index=["yy"],columns="mm",aggfunc=np.sum)
    
    axs[0].set_title('obs.value_mm',y=1.015)
    axs[1].set_title('sim.value_mm',y=1.015)
    axs[2].set_title('sim.value_mm',y=1.015)
    im_left= axs[0].imshow(heatmap_obs, cmap = 'hot',interpolation='nearest', vmin = 0, vmax =700)
    im_right= axs[1].imshow(heatmap_one, cmap = 'hot',interpolation='nearest', vmin = 0, vmax =700)
    im_right= axs[2].imshow(heatmap_lstm, cmap = 'hot',interpolation='nearest', vmin = 0, vmax =700)
    fig.colorbar(im_left,ax=axs[0])
    fig.colorbar(im_right,ax=axs[1])
    fig.colorbar(im_right,ax=axs[2])
    xx_label=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axs[0].set_xticks(range(len(heatmap_obs.columns)), xx_label, rotation = 45)
    axs[0].set_yticks(range(len(heatmap_obs)), heatmap_obs.index, rotation = 0)
    axs[1].set_xticks(range(len(heatmap_one.columns)), xx_label, rotation = 45)
    axs[1].set_yticks(range(len(heatmap_one)), heatmap_one.index, rotation = 0)
    axs[2].set_xticks(range(len(heatmap_lstm.columns)), xx_label, rotation = 45)
    axs[2].set_yticks(range(len(heatmap_lstm)), heatmap_lstm.index, rotation = 0)
    axs[0].set_aspect('auto')
    axs[1].set_aspect('auto')
    axs[2].set_aspect('auto')
    plt.tight_layout()
    plt.show()


def show_graph_runoff_ratio(raw):
    data=raw
    data['yyyy']=data['date'].dt.year
    data=data.groupby('yyyy')['rf','obs_mm','sim_mm_one','sim_mm_lstm'].sum()
    data['ratio_obs']=data['obs_mm']/data['rf']*100
    data['ratio_one']=data['sim_mm_one']/data['rf']*100
    data['ratio_lstm']=data['sim_mm_lstm']/data['rf']*100
    data=np.round(data[['ratio_obs','ratio_one','ratio_lstm']],2)

    fig, ax = plt.subplots(ncols=1, figsize=(8, 7), constrained_layout=True,sharex=True, sharey=True)
    sns.regplot(x="ratio_obs", y="ratio_one", data=data, ax=ax, color="k",ci=None,fit_reg=True,
                scatter_kws={"fc":"black", "ec":"black", "s":30},
                line_kws ={"lw":1,},label="ONE model")
    sns.regplot(x="ratio_obs", y="ratio_lstm", data=data, ax=ax, color="blue",ci=None,fit_reg=True,
                scatter_kws={"fc":"royalblue", "ec":"darkblue", "s":30},
                line_kws ={"lw":1,},label="LSTM model")
    x=range(0,100)
    sns.lineplot(x=x, y=x, color='gray', linestyle='--')
    ax.set_ylim(35,85 )
    ax.set_xlim(35,85 )
    ax.legend()
    plt.xlabel('Runoff Ratio Pct.(Obs)')
    plt.ylabel('Runoff Ratio Pct.(model)')
    plt.tight_layout()
    plt.show()

##########################################

station='yd'
ONE = pd.read_excel("D:/OneDrive/ONE_model/results/ONE_output_"+station+".xlsx")
ONE_result=ONE[['date','rf', 'obs_mm', 'obs_cms','sim_mm','sim_cms']]
ONE_result.rename(columns = {"sim_mm":"sim_mm_one","sim_cms":"sim_cms_one"},inplace=True)
ONE_result.set_index('date')
print(ONE_result)



LSTM = pd.read_excel("D:/OneDrive/ONE_model/results/lstm_output_"+station+".xlsx")
LSTM_result=LSTM[['date','obs_cms','sim_mm','sim_cms']]
LSTM_result.rename(columns = {'sim_mm':'sim_mm_lstm','sim_cms':'sim_cms_lstm'},inplace=True)
LSTM_result.set_index('date')
print(LSTM_result)

ONE_LSTM=pd.merge(ONE_result, LSTM_result,on='date', how='left')
ONE_LSTM=ONE_LSTM[['date','rf', 'obs_mm', 'obs_cms_x','sim_mm_one','sim_cms_one','sim_mm_lstm','sim_cms_lstm']]
ONE_LSTM=np.round(ONE_LSTM,3)
ONE_LSTM.rename(columns = {'obs_cms_x':'obs_cms'},inplace=True)
ONE_LSTM.to_excel("D:/OneDrive/ONE_model/results/ONE_LSTM_"+station+".xlsx")

ONE_LSTM['yyyy'] = ONE_LSTM['date'].dt.year
grp=ONE_LSTM.groupby('yyyy')['rf','obs_mm','sim_mm_one','sim_mm_lstm'].sum()
print(grp)


stt='2002-01-01'
ett='2012-12-31'
raw=ONE_LSTM
show_graph_day_one_lstm(raw,stt,ett)

stt='2013-01-01'
ett='2021-12-31'
show_graph_day_one_lstm(raw,stt,ett)

stt='2002-01-01'
ett='2021-12-31'
raw=ONE_LSTM
show_graph_line_one_lstm(raw, stt,ett)

  
    
raw = ONE_LSTM
show_graph_heatmap_one_lstm(raw)

raw = ONE_LSTM
show_graph_runoff_ratio(raw)

