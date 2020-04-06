# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:09:10 2020

@author: ycbfr

[project functions]
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels import PanelOLS
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

from statsmodels.stats.outliers_influence import variance_inflation_factor


def vif_calculation(df_vars):
    """calculate variance inflation factors
    """
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(df_vars.values, i) for i in range(df_vars.shape[1])]
    vif["features"] = df_vars.columns
    return vif.round(2)

def weight_mean(df, weight_col, value_col):
    weight = df[weight_col] / df[weight_col].sum()
    return df[value_col].dot(weight)

def align_time_frame(time_range, time_index):
    """
    Find the end of each month/year of a time index
    
    Parameters
    ----------
    time_range : pd.DatetimeIndex
        a list of time stamps of the end of natural month/year, 
        
    time_index : pd.DatetimeIndex
        a list of all time stamps appear
    
    Returns
    -------
    time_list : pd.DatetimeIndex
        a list of the end of each month/year of a time index
    """
    time_list = list(time_range)
    for i in range(0, len(time_list)):
        while time_list[i] not in time_index: 
            time_list[i] = time_list[i] - pd.Timedelta(1, 'D')
    time_range_aligned = pd.DatetimeIndex(time_list)
    return time_range_aligned

def back_test_linear(df, DATE_START, DATE_END, y_var, var_list, return_var='excess_return_f', NUM_BUCKET=10):
    month_index = pd.date_range(DATE_START, DATE_END, freq='M')
    month_index = align_time_frame(month_index, pd.Index(df['date'].unique()))
    market_median = df.groupby('date')['excess_return'].median()
    
    df_return = pd.DataFrame(index=month_index, columns=range(1, NUM_BUCKET+1))
    df_return['market_median'] = market_median.loc[DATE_START:DATE_END].values
    df_conf = pd.DataFrame(index=month_index, columns=['TP', 'FP', 'FN', 'TN'])
    df_mse = pd.DataFrame(index=month_index, columns=['MSE', 'fund_num'])
    
    fund_reg_all = df[(df['date']<DATE_END)&(df['date']>=DATE_START)]
    data_reg_all = fund_reg_all[['date', 'fundno']+var_list+[y_var]]
    data_reg_all = data_reg_all.set_index(['fundno', 'date'])
    X_reg_all = data_reg_all[var_list]
    X_reg_all = sm.add_constant(X_reg_all)
    y_reg_all = data_reg_all[y_var]
    mod_all = PanelOLS(y_reg_all, X_reg_all, time_effects=True)
    reg_res_all = mod_all.fit()
    y_pred_all = reg_res_all.predict(X_reg_all)
    MSE_all = mean_squared_error(y_reg_all, y_pred_all)
    
    for i in range(len(month_index)-1):
        date = month_index[i]
        date_f = month_index[i+1]
        fund_reg = df[df['date']<date]
        data_reg = fund_reg[['date', 'fundno']+var_list+[y_var]]
        data_reg = data_reg.set_index(['fundno', 'date'])
        X_reg = data_reg[var_list]
        X_reg = sm.add_constant(X_reg)
        y_reg = data_reg[y_var]
        mod = PanelOLS(y_reg, X_reg, time_effects=True)
        reg_res = mod.fit()
        fund_pred = df[df['date']==date]
        data_pred = fund_pred[['date', 'fundno']+var_list+[y_var]]
        y_act = data_pred[['fundno', y_var]]
        y_act = y_act.set_index('fundno')
        y_act_bin = pd.Series(index=y_act.index)
        y_act_bin.loc[y_act[y_act[y_var]>0].index] = 1
        y_act_bin.loc[y_act[y_act[y_var]<=0].index] = 0
        data_pred = data_pred.set_index(['fundno', 'date'])
        X_pred = data_pred[var_list]
        X_pred = sm.add_constant(X_pred)
        y_pred = reg_res.predict(X_pred)
        y_pred = y_pred.reset_index(level=1).drop(columns='date')
        
        df_mse.loc[date_f, 'MSE'] = mean_squared_error(y_act[y_var].values, y_pred['predictions'].values)
        df_mse.loc[date_f, 'fund_num'] = len(y_act[y_var])
        
        y_pred_bin = pd.Series(index=y_pred.index)
        y_pred_bin.loc[y_pred[y_pred['predictions']>0].index] = 1
        y_pred_bin.loc[y_pred[y_pred['predictions']<=0].index] = 0
        y_pred['decile'] = 11 - pd.qcut(y_pred['predictions'], 10, labels=range(1,NUM_BUCKET+1)).astype(int)
        for j in range(1,NUM_BUCKET+1):
            decile_j = y_pred[y_pred['decile']==j].index
            df_return.loc[date_f, j] = weight_mean(fund_pred[fund_pred['fundno'].isin(decile_j)][['mtna', return_var]], 'mtna', return_var)
            date_decile_index = fund_pred[fund_pred['fundno'].isin(decile_j)].index
            df.loc[date_decile_index, 'decile'] = j
        df_conf.loc[date_f] = confusion_matrix(y_act_bin, y_pred_bin, labels=[1,0]).reshape((1,4))
    df_return.iloc[0,:] = 1
    MSE_sample = weight_mean(df_mse.iloc[1:,], 'fund_num', 'MSE')
    return df_return, df_conf, df_mse, (MSE_all, MSE_sample), df

def back_test_logistic(df, DATE_START, DATE_END, y_var, var_list, return_var='excess_return_f', NUM_BUCKET=10):
    month_index = pd.date_range(DATE_START, DATE_END, freq='M')
    month_index = align_time_frame(month_index, pd.Index(df['date'].unique()))
    market_median = df.groupby('date')['excess_return'].median()
    
    df_return = pd.DataFrame(index=month_index, columns=range(1, NUM_BUCKET+1))
    df_return['market_median'] = market_median.loc[DATE_START:DATE_END].values
    df_conf = pd.DataFrame(index=month_index, columns=['TP', 'FP', 'FN', 'TN'])
    
    for i in range(len(month_index)-1):
        date = month_index[i]
        date_f = month_index[i+1]
        fund_reg = df[df['date']<date]
        data_reg = fund_reg[['date', 'fundno']+var_list+[y_var]]
        data_reg.loc[data_reg[y_var]>0, y_var] = 1
        data_reg.loc[data_reg[y_var]<=0, y_var] = 0
        data_reg[var_list] = (data_reg[var_list] - data_reg[var_list].mean()) / data_reg[var_list].std()
        X_reg = data_reg[var_list]
        X_reg = sm.add_constant(X_reg)
        y_reg = data_reg[y_var]
        mod = sm.Logit(y_reg, X_reg)
        reg_res = mod.fit()
        
        fund_pred = df[df['date']==date]
        data_pred = fund_pred[['date', 'fundno']+var_list+[y_var]]
        data_pred.loc[data_pred[y_var]>0, y_var] = 1
        data_pred.loc[data_pred[y_var]<=0, y_var] = 0
        data_pred[var_list] = (data_pred[var_list] - data_pred[var_list].mean()) / data_pred[var_list].std()
        y_act = data_pred[['fundno', y_var]]
        y_act = y_act.set_index('fundno')

        X_pred = data_pred[var_list]
        X_pred = sm.add_constant(X_pred)
        y_pred = reg_res.predict(X_pred)
        y_pred = pd.DataFrame(y_pred).reset_index()
        y_pred.rename(columns={'index':'fundno', 0:'predictions'}, inplace=True)
        y_pred['fundno'] = data_pred['fundno'].values
        y_pred.set_index('fundno', inplace=True)
        
        y_pred['decile'] = 11 - pd.qcut(y_pred['predictions'], 10, labels=range(1,NUM_BUCKET+1)).astype(int)
        
        y_pred_bin = pd.Series(index=y_pred.index)
        y_pred_bin.loc[y_pred[y_pred['predictions']>y_pred['predictions'].median()].index] = 1
        y_pred_bin.loc[y_pred[y_pred['predictions']<=y_pred['predictions'].median()].index] = 0
        
        for j in range(1,NUM_BUCKET+1):
            decile_j = y_pred[y_pred['decile']==j].index
            df_return.loc[date_f, j] = weight_mean(fund_pred[fund_pred['fundno'].isin(decile_j)][['mtna', return_var]], 'mtna', return_var)
            date_decile_index = fund_pred[fund_pred['fundno'].isin(decile_j)].index
            df.loc[date_decile_index, 'decile'] = j
        df_conf.loc[date_f] = confusion_matrix(y_act, y_pred_bin, labels=[1,0]).reshape((1,4))
    df_return.iloc[0,:] = 1
    return df_return, df_conf, df

def conf_mat_comp(df_conf): 
    dis_model = df_conf.sum()
    df_comp = pd.DataFrame(index=['Accuracy', 'Precision', 'Recall', 'F1'], columns=['ALL_POSITIVE', 'MODEL'])
    df_comp.loc[:, 'ALL_POSITIVE'] = 0.5
    df_comp.loc['Accuracy', 'MODEL'] = (dis_model.loc['TP'] + dis_model.loc['TN'])/ dis_model.sum()
    df_comp.loc['Precision', 'MODEL'] = dis_model.loc['TP'] / (dis_model.loc['TP'] + dis_model.loc['FP'])
    df_comp.loc['Recall', 'MODEL'] = dis_model.loc['TP'] / (dis_model.loc['TP'] + dis_model.loc['FN'])
    df_comp.loc['F1', 'MODEL'] = (2 * df_comp.loc['Precision', 'MODEL'] * df_comp.loc['Recall', 'MODEL']) / (df_comp.loc['Precision', 'MODEL'] + df_comp.loc['Recall', 'MODEL'])
    return df_comp

def mdd(series):
    """Calculates maximum drawdown of return series, ordered by columns in a DataFrame. Returns 0 for series without any data.
    """
    series_start_zero = pd.Series(index=range(len(series) + 1))
    series_start_zero.iloc[1:] = series.values
    # the first row of df_start_zero is np.nan -> to convert to zero, together with other missing values (due to time series starting 
    # after start sample)
    series_start_zero.iloc[0] = 0
    cum_df = series_start_zero.cumsum()
    roll_max = cum_df.cummax()
    drawdown = cum_df - roll_max
    mdd = - drawdown.min()
    return mdd


INDEX_COL = ['Annualised Return', 'Annualized Standard Deviation', 'Annualised IR', 'Maximum Drawdown Total', 'Maximum Drawdown Excess', 'Largest Monthly Gain', 'Largest Monthly Loss', 'Avg rolling 3 Yr IR', 'Avg rolling 5 Yr IR']
def perform_eval(port_return, bench_return, trade_freq): 
    excess_return = port_return - bench_return
    perform_mat = pd.Series(index=INDEX_COL)
    perform_mat['Annualised Return'] = port_return.mean() * trade_freq
    perform_mat['Annualized Standard Deviation'] = port_return.std() * np.sqrt(trade_freq)
    perform_mat['Annualised IR'] = excess_return.mean() / excess_return.std() * np.sqrt(trade_freq)
    perform_mat['Maximum Drawdown Total'] = mdd(port_return)
    perform_mat['Maximum Drawdown Excess'] = mdd(excess_return)
    perform_mat['Largest Monthly Gain'] = port_return.max()
    perform_mat['Largest Monthly Loss'] = port_return.min()
    perform_mat['Avg rolling 3 Yr IR'] = (excess_return.rolling(36).mean() / excess_return.rolling(36).std()).mean() * np.sqrt(trade_freq)
    perform_mat['Avg rolling 5 Yr IR'] = (excess_return.rolling(60).mean() / excess_return.rolling(60).std()).mean() * np.sqrt(trade_freq)
    perform_mat['Beta'] = bench_return.corr(port_return) * port_return.std() / bench_return.std()
    return perform_mat

def perform_eval_df(df, num_bucket=10):
    perform_df = pd.DataFrame(index=range(1, num_bucket+1), columns=INDEX_COL)
    for i in range(1, num_bucket+1):
        port_return = df[i].astype('float64')
        bench_return = df['market_median'].astype('float64')
        perform_df.loc[i, :] = perform_eval(port_return, bench_return, 12)    
    return perform_df

def decile_description(df, var_list):
    df_decile = pd.DataFrame(columns=var_list, index=df['decile'].unique())
    df_decile.sort_index(inplace=True)
    for var in var_list:
        if var == 'mtna':
            df_decile[var] = df.groupby('decile')[var].median()
        else:
            df_decile[var] = df.groupby('decile').apply(weight_mean, 'mtna', var)
    return df_decile

FACTOR_LIST = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
def FF_reg(df_return, df_FF, num_bucket=10):
    p_list = ['p_' + s for s in ['alpha']+FACTOR_LIST]
    df_reg = pd.DataFrame(columns=['alpha']+FACTOR_LIST+p_list, index=range(1, num_bucket+1))
    X_reg = df_FF[FACTOR_LIST] / 100
    X_reg = sm.add_constant(X_reg)
    for i in range(1, num_bucket+1):
        y_reg = df_return[i].iloc[1:]
        y_reg = y_reg.astype(float)
        fit = sm.OLS(y_reg.values, X_reg.values).fit()
        df_reg.loc[i, ['alpha']+FACTOR_LIST] = fit.params
        df_reg.loc[i, p_list] = fit.pvalues       
    return df_reg