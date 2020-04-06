# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:49:43 2020

@author: ycbfr
"""
import pandas as pd
import numpy as np
import project_functions as pf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import regularizers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data
df_class = pd.read_csv(r"..\Data\Settings\fund_class.csv", index_col=0)
df = pd.read_csv(r"..\Data\monthly_return_final.csv", parse_dates=['caldt'])
df_info = pd.read_csv(r"..\Data\raw_data\fund_info_filtered.csv", usecols=['crsp_fundno', 'lipper_class'])

# Rename columns
df.rename(columns={'caldt':'date', 'crsp_fundno':'fundno'}, inplace=True)

# Seperate equity fund and mix fund
equity_list = df_info[df_info['lipper_class'].isin(df_class['Equity'])]['crsp_fundno'].unique()
mix_list = df_info[df_info['lipper_class'].isin(df_class['Mixed'])]['crsp_fundno'].unique()

df_equity = df[df['fundno'].isin(equity_list)]
df_mix = df[df['fundno'].isin(mix_list)]

BETAS = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
#NUM_LIST = ['mtna', 'excess_market', 'vol', 'alpha', 'r2', 'exp_ratio', 'turn_ratio']
NUM_LIST = ['mtna', 'vol', 'alpha', 'r2', 'exp_ratio', 'turn_ratio']
VAR_LIST = NUM_LIST + BETAS
RETURN_VAR = 'excess_return_f'

df_equity = df_equity.dropna(subset=VAR_LIST+[RETURN_VAR])
df_mix = df_mix.dropna(subset=VAR_LIST+[RETURN_VAR])


# Back Test
DATE_START = pd.Timestamp(2003, 12, 31)
DATE_INTER = pd.Timestamp(2004, 6, 30)
DATE_END = pd.Timestamp(2019, 8, 31)
NUM_BUCKET = 10

FF_monthly = pd.read_csv(r"..\Data\FF_monthly.csv", skiprows=3, index_col=0)
FF_monthly.rename(columns={'Mkt-RF':'MKT'}, inplace=True)
df_FF = FF_monthly.loc[DATE_START.year*100+DATE_START.month+1:DATE_END.year*100+DATE_END.month]

EPOCHS = 5
NUM_BUCKET=10
Y_VAR = RETURN_VAR




for name, df_sub in {'equity':df_equity, 'mix':df_mix}.items():
    model = Sequential()
    model.add(LSTM(10, 
                   input_shape=(None, len(VAR_LIST)),
                   use_bias = True,
                   dropout=0.1, 
                   recurrent_dropout=0.1, 
                   kernel_regularizer=regularizers.l1_l2(l1=0.03, l2=0.03)))
    model.add(Dense(1, activation='linear', input_dim=10))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    month_index = pd.date_range(DATE_START, DATE_END, freq='M')
    month_index = pf.align_time_frame(month_index, pd.Index(df_sub['date'].unique()))
    market_median = df_sub.groupby('date')['excess_return'].median()    
    df_return = pd.DataFrame(index=month_index, columns=range(1, NUM_BUCKET+1))
    df_return['market_median'] = market_median.loc[DATE_START:DATE_END].values
    df_conf = pd.DataFrame(index=month_index, columns=['TP', 'FP', 'FN', 'TN'])
    df_mse = pd.DataFrame(index=month_index, columns=['MSE', 'fund_num'])
    
    for i in range(len(month_index)-1):        
        date = month_index[i]
        date_f = month_index[i+1]
        fund_reg = df_sub[df_sub['date']<date]
        date_s = fund_reg['date'].min()
        date_e = fund_reg['date'].max()
        fund_reg = fund_reg[fund_reg['fundno'].isin(df_sub[df_sub['date']==date_e]['fundno'])]
        fund_reg[VAR_LIST] = (fund_reg[VAR_LIST] - fund_reg[VAR_LIST].mean()) / fund_reg[VAR_LIST].std()
        X_reg = fund_reg[['fundno', 'date']+VAR_LIST].set_index(['fundno', 'date'], drop=True).to_xarray().to_array().fillna(0).values
        X_reg = np.swapaxes(X_reg, 0, 1)
        X_reg = np.swapaxes(X_reg, 1, 2)  
        y_reg = fund_reg[fund_reg['date']==date_e][['fundno', Y_VAR]].set_index('fundno').values
        if date <= DATE_INTER:
            model.fit(X_reg, y_reg, epochs=EPOCHS * 4)
        else:
            model.fit(X_reg, y_reg, epochs=EPOCHS)        
        date_s = fund_reg['date'].min()    
        fund_pred = df_sub[df_sub['date']<=date]
        fund_pred = fund_pred[fund_pred['fundno'].isin(df_sub[df_sub['date']==date]['fundno'])]
        fund_pred[VAR_LIST] = (fund_pred[VAR_LIST] - fund_pred[VAR_LIST].mean()) / fund_pred[VAR_LIST].std()
        y_act = fund_pred[fund_pred['date']==date][['fundno', Y_VAR]].set_index('fundno')
        y_act_bin = pd.Series(index=y_act.index)
        y_act_bin.loc[y_act[y_act[Y_VAR]>0].index] = 1
        y_act_bin.loc[y_act[y_act[Y_VAR]<=0].index] = 0
        
        X_pred = fund_pred[['fundno', 'date']+VAR_LIST].set_index(['fundno', 'date'], drop=True).to_xarray().to_array().fillna(0).values
        X_pred = np.swapaxes(X_pred, 0, 1)
        X_pred = np.swapaxes(X_pred, 1, 2)
        y_pred = model.predict(X_pred)
        y_pred = pd.DataFrame(y_pred, index=y_act.index)
        y_pred.rename(columns={0:'predictions'}, inplace=True)
        
        df_mse.loc[date_f, 'MSE'] = mean_squared_error(y_act[Y_VAR].values, y_pred['predictions'].values)
        df_mse.loc[date_f, 'fund_num'] = len(y_act[Y_VAR])
        
        y_pred['decile'] = 11 - pd.qcut(y_pred['predictions'], 10, labels=range(1,NUM_BUCKET+1)).astype(int)
        
        y_pred_bin = pd.Series(index=y_pred.index)
        y_pred_bin.loc[y_pred[y_pred['predictions']>y_pred['predictions'].median()].index] = 1
        y_pred_bin.loc[y_pred[y_pred['predictions']<=y_pred['predictions'].median()].index] = 0
        for j in range(1,NUM_BUCKET+1):    
            decile_j = y_pred[y_pred['decile']==j].index
            fund_pred_spot = df_sub[df_sub['date']==date]            
            df_return.loc[date_f, j] = pf.weight_mean(fund_pred_spot[fund_pred_spot['fundno'].isin(decile_j)][['mtna', RETURN_VAR]], 'mtna', RETURN_VAR)
            date_decile_index = fund_pred_spot[fund_pred_spot['fundno'].isin(decile_j)].index
            df_sub.loc[date_decile_index, 'decile'] = j
            df_conf.loc[date_f] = confusion_matrix(y_act_bin, y_pred_bin, labels=[1,0]).reshape((1,4))
    
    df_return.iloc[0,:] = 1
    df_return.cumsum().plot(figsize=(20, 10))
    plt.title('Portfolio Performance_' + name)
    plt.savefig(r"..\Data\process_store\LSTM\port_perf_"+name+".png")
    df_return.to_csv(r"..\Data\process_store\LSTM\port_return_"+name+".csv", index_col=0)
    df_conf.to_csv(r"..\Data\process_store\LSTM\port_conf_"+name+".csv")    
    df_mse.to_csv(r"..\Data\process_store\LSTM\mse_"+name+".csv")
    df_eval = pf.perform_eval_df(df_return.iloc[1:])
    df_eval.to_csv(r"..\Data\process_store\LSTM\port_eval_"+name+".csv")
    df_sub.dropna(subset=['decile'], inplace=True)
    df_sub['vol'] = df_sub['vol'] * np.sqrt(252)
    df_decile = pf.decile_description(df_sub, NUM_LIST)
    df_decile['excess_market'] = df_decile['excess_market'] * 12
    df_decile['alpha'] = df_decile['alpha'] * 252
    df_decile.to_csv(r"..\Data\process_store\LSTM\decile_desc_"+name+".csv")
    df_decile_FF = pf.FF_reg(df_return, df_FF)
    df_decile_FF['alpha'] = df_decile_FF['alpha'] * 12
    df_decile_FF.to_csv(r"..\Data\process_store\LSTM\decile_desc_FF_"+name+".csv")    