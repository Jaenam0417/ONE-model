# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from permetrics.regression import RegressionMetric
#https://github.com/thieu1995/permetrics


##first loading############################################################
#st.title("ONE model")
st.subheader('ONE (One-parameter New Exponential) model')
st.info('Introduction to ONE model')
st.write("**O**ne-parameter **N**ew **E**xponential model was developed to simulate daily rainfall-runoff using a single parameter depending on soil water storage in a specific watershed. The model uses one parameter to estimate daily runoff based on an exponential function, which is a nonlinear relationship. The hydrological factors of the watershed are divided into rainfall, evaporation, and runoff. The runoff was designed to be affected by daily soil retention to minimize the parameters of the hydrological model.")
##image add
image = Image.open('./temp/ONE_model_concept.PNG')
#image="https://raw.githubusercontent.com/Jaenam0417/ONE_model/main/temp/ONE_model_concept.PNG?token=GHSAT0AAAAAAB4QYYNCK6MVDSIOYFQ7Y7P4Y5AEFBA"
st.image(image, caption='A schematic diagram for the ONE model')
st.write("Figure shows a schematic of the ONE model. The boundary of the box represents the watershed, and the watershed storage condition increases with the increase in the amount of rainfall in the watershed. This can also be explained by a simple hydrological structure wherein the watershed storage decreases due to evaporation and runoff from the watershed. In this case, when the watershed is dry, soil water storage reaches a limit, and watershed evaporation does not occur. However, runoff is allowed to occur continuously according to soil retention even when the watershed is dry. The ONE model uses rainfall and potential evapotranspiration as the input data and is designed to use a single parameter so that the user can easily determine it, considering that a single parameter (w) is dependent on the local data.")


##sidebar##################################################################
st.sidebar.title("ONE model")
st.sidebar.success('Model setting')

selected_target = st.sidebar.radio("Please select a study area", ('Soyanggang_Dam', 'Youngdam_Dam'),horizontal=False, key='target')
if selected_target=='Soyanggang_Dam':
    station='sy'
    single_para_w=2.5
if selected_target=='Youngdam_Dam':
    station='yd'
    single_para_w=2.96

data_year = st.sidebar.slider('Please select a range of analysis period', 2002, 2021, (2002, 2021))
st.sidebar.write('Selected period:', data_year)
st.sidebar.write(' ')

single_para_w = st.sidebar.slider('Please select a paramater (w)', 0.0, 10.0, (single_para_w), key='single_para_w')
st.sidebar.write('Selected a single paramater value (w):', single_para_w)
st.sidebar.write(' ')

st.sidebar.success('Input data')
if selected_target=='Soyanggang_Dam':
    uploaded_file = pd.read_csv('./datasets/sy.csv')   #url = "https://raw.githubusercontent.com/Jaenam0417/ONE_model/main/datasets/sy.csv?token=GHSAT0AAAAAAB4QYYNDOEYGXWMMNVGT5LF4Y5AAPFA"
if selected_target=='Youngdam_Dam':
    uploaded_file = pd.read_csv('./datasets/yd.csv')   #url = "https://raw.githubusercontent.com/Jaenam0417/ONE_model/main/datasets/sy.csv?token=GHSAT0AAAAAAB4QYYNDOEYGXWMMNVGT5LF4Y5AAPFA"

uploaded_file=uploaded_file[['date', 'rf', 'etp', 'inflow']]
uploaded_file.rename(columns = {'date':'Date','rf':'Rainfall', 'etp':'ETp', 'inflow':'Inflow'},inplace=True)
T=uploaded_file.style.format({'Rainfall' : '{:.2f}', 'ETp' : '{:.2f}', 'Inflow' : '{:.2f}'})
st.sidebar.dataframe(T, 300,300)

##sidebar##################################################################

def run_setting(station):
    global area,last_day_of_the_month
    last_day_of_the_month = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    if station=="sy":
        area=2703
    if station=="yd":
        area=930
    return area, last_day_of_the_month

def data_read(syear, eyear):
    global rf,ep, sw, et, qq, obs, df, F_name, raw
    #raw = pd.read_excel(F_name+".xlsx",sheet_name="input")
    #raw = pd.read_csv('"'+ D:\\OneDrive\\ONE_model\\datasets\\yd.csv')
    url='./datasets/'+station+'.csv'
    raw = pd.read_csv(url)
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
    return raw

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
    global rf, ep, sw, et, qq, obs, df, area
    df['rf'] = rf.iloc[:,0]
    df['sw'] = sw.iloc[:,0]
    df['et'] = et.iloc[:,0]
    df['qq'] = qq.iloc[:,0] 
    
    df['sim_mm'] = df['qq']
    df['sim_cms'] = df['sim_mm']*area/86.4
    df['obs_cms'] = df['inflow']
    df['obs_mm'] = df['obs_cms']*86.4/area
    return df

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

def show_graph_day(yy1, yy2,sty,ety):
    global fig
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
    return fig

def show_graph_line(xx, yy):
    global fig
    fig, (ax1) = plt.subplots(1, 1,figsize=(4.5,3.0))
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
    return fig

def show_graph_heatmap(xx):
    global fig
    output=xx
    fig, axs = plt.subplots(1, 2, figsize=(8,5))
    output['mm']=output['date'].dt.month
    output['yy']=output['date'].dt.year
    output=output[['yy', 'mm', 'sim_mm','obs_mm']]
    heatmap_obs = pd.pivot_table(output,values="obs_mm",index=["yy"],columns="mm",aggfunc=np.sum)
    heatmap_sim = pd.pivot_table(output,values="sim_mm",index=["yy"],columns="mm",aggfunc=np.sum)
    
    axs[0].set_title('obs.value_mm')
    axs[1].set_title('sim.value_mm')
    im_left= axs[0].imshow(heatmap_obs, cmap = 'hot',interpolation='nearest', vmin = 0, vmax =700)
    im_right= axs[1].imshow(heatmap_sim, cmap = 'hot',interpolation='nearest', vmin = 0, vmax =700)
    fig.colorbar(im_left,ax=axs[0])
    fig.colorbar(im_right,ax=axs[1])
    xx_label=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axs[0].set_xticks(range(len(heatmap_obs.columns)), xx_label, rotation = 45)
    axs[0].set_yticks(range(len(heatmap_obs)), heatmap_sim.index, rotation = 0)
    axs[1].set_xticks(range(len(heatmap_obs.columns)), xx_label, rotation = 45)
    axs[1].set_yticks(range(len(heatmap_obs)), heatmap_sim.index, rotation = 0)
    axs[0].set_aspect('auto')
    axs[1].set_aspect('auto')
    plt.tight_layout()
    plt.show()
    return fig

########streamlit_code########################################################
station=station
syear=data_year[0]
eyear=data_year[1]
para_w= single_para_w

run_setting(station)
data_read(syear, eyear)
calcaul(syear, eyear, para_w)
simulation_result()
result_estimate(df['obs_mm'],df['sim_mm'])
ONE=df[['date','rf','sw','sim_mm','sim_cms','obs_mm','obs_cms']]

del st.session_state['single_para_w']
###
stt=str(syear)+'-01-01'  ##'2002-01-01'
ett=str(eyear)+'-12-31'  ##'2021-12-31'
tmp =df.loc[ONE['date'].between(stt, ett)]

st.info('ONE model simulation & Evaluation results')
st.markdown('**# Runoff simulation**')

###Daily
show_graph_day(ONE['obs_cms'],ONE['sim_cms'],stt,ett)
st.write(fig)

result_estimate(tmp['obs_mm'],tmp['sim_mm'])
st.markdown('**# Evaluation index**')
col1, col2, col3, col4 = st.columns(4)
col1.metric("Nash–Sutcliffe efficiency", str(round(NSE,2)))
col2.metric("Coefficient of determination", str(round(R2,2)))
col4.metric("Percent bias (%)", str(round(PBIAS,2)))

result_estimate(tmp['obs_cms'],tmp['sim_cms'])
col3.metric("Root Mean Square Error (cms)", str(round(RMSE,2)))


st.markdown('**# Daily scatter plot**')
show_graph_line(tmp['obs_mm'],tmp['sim_mm'])
st.write(fig)

###Monthly
st.markdown('**# Monthly heatmap**')
show_graph_heatmap(ONE)
st.write(fig)

###Yearly
st.markdown('**# Yearly runoff ratio**')
#ONE_yy=df.groupby('yyyy')['rf','et','sim_mm','obs_mm'].sum()
#ONE_yy['ratio_obs']=ONE_yy['obs_mm']/ONE_yy['rf']*100
#ONE_yy['ratio_one']=ONE_yy['sim_mm']/ONE_yy['rf']*100
#ONE_yy=ONE_yy[['rf','et','obs_mm','sim_mm','ratio_obs','ratio_one']]
#ONE_yy.rename(columns = {'rf':'Rainfall','et':'ET', 'obs_mm':'Obs.value(mm)', 'sim_mm':'Sim.value(mm)', 'ratio_obs':'Obs.ratio(%)', 'ratio_one':'Sim.ratio(%)'},inplace=True)
#TT=ONE_yy.style.format({'Rainfall' : '{:.2f}', 'ET' : '{:.2f}', 'Obs.value(mm)' : '{:.2f}', 'Sim.value(mm)' : '{:.2f}', 'Obs.ratio(%)' : '{:.2f}', 'Sim.ratio(%)' : '{:.2f}'})
#st.dataframe(TT, 1000, 300)
                   

