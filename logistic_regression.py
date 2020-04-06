# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:25:06 2020

@author: ycbfr

[logistic regression]
"""
import pandas as pd
import numpy as np
import project_functions as pf
import statsmodels.api as sm
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
NUM_LIST = ['mtna', 'excess_market', 'vol', 'alpha', 'r2', 'exp_ratio', 'turn_ratio']
VAR_LIST = NUM_LIST + BETAS
RETURN_VAR = 'excess_return_f'
Y_VAR = 'excess_return_f_binary'

df_equity = df_equity.dropna(subset=VAR_LIST+[RETURN_VAR])
df_mix = df_mix.dropna(subset=VAR_LIST+[RETURN_VAR])

df_equity[VAR_LIST] = (df_equity[VAR_LIST] - df_equity[VAR_LIST].mean()) / df_equity[VAR_LIST].std()
df_mix[VAR_LIST] = (df_mix[VAR_LIST] - df_mix[VAR_LIST].mean()) / df_mix[VAR_LIST].std()

df_equity.loc[df_equity[RETURN_VAR]>0, Y_VAR] = 1
df_equity.loc[df_equity[RETURN_VAR]<=0, Y_VAR] = 0

df_mix.loc[df_mix[RETURN_VAR]>0, Y_VAR] = 1
df_mix.loc[df_mix[RETURN_VAR]<=0, Y_VAR] = 0

with open(r"..\Data\process_store\logistic\reg_results.txt", 'w') as reg_file:
    for df_sub in [df_equity, df_mix]:
        logit = sm.Logit(df_sub[Y_VAR], sm.add_constant(df_sub[VAR_LIST]))
        log_res = logit.fit()
        reg_file.write(log_res.summary().as_latex())
        reg_file.write('\n\n')

# Reload data
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
VAR_LIST = ['mtna', 'excess_market', 'vol', 'alpha', 'r2', 'exp_ratio', 'turn_ratio'] + BETAS
Y_VAR = 'excess_market_f'

df_equity = df_equity.dropna(subset=VAR_LIST+[Y_VAR])
df_mix = df_mix.dropna(subset=VAR_LIST+[Y_VAR])

# Back Test
DATE_START = pd.Timestamp(2003, 12, 31)
DATE_END = pd.Timestamp(2019, 8, 31)

FF_monthly = pd.read_csv(r"..\Data\FF_monthly.csv", skiprows=3, index_col=0)
FF_monthly.rename(columns={'Mkt-RF':'MKT'}, inplace=True)
df_FF = FF_monthly.loc[DATE_START.year*100+DATE_START.month+1:DATE_END.year*100+DATE_END.month]


for name, df_sub in {'equity':df_equity, 'mix':df_mix}.items():
    df_return, df_conf, df_sub = pf.back_test_logistic(df_sub, DATE_START, DATE_END, Y_VAR, VAR_LIST)
    df_return.to_csv(r"..\Data\process_store\logistic\port_return_"+name+".csv")
    df_conf.to_csv(r"..\Data\process_store\logistic\port_conf_"+name+".csv")        
    df_comp_sub = pf.conf_mat_comp(df_conf)
    df_comp_sub.to_csv(r"..\Data\process_store\logistic\conf_"+name+".csv")
    df_return.cumsum().plot(figsize=(20, 10))
    plt.title('Portfolio Performance_' + name)
    plt.savefig(r"..\Data\process_store\logistic\port_perf "+name+".png")
    df_eval = pf.perform_eval_df(df_return.iloc[1:])
    df_eval.to_csv(r"..\Data\process_store\logistic\port_eval_"+name+".csv")
    df_sub.dropna(subset=['decile'], inplace=True)
    df_sub['vol'] = df_sub['vol'] * np.sqrt(252)
    df_decile = pf.decile_description(df_sub, NUM_LIST)
    df_decile['excess_market'] = df_decile['excess_market'] * 12
    df_decile['alpha'] = df_decile['alpha'] * 252
    df_decile.to_csv(r"..\Data\process_store\logistic\decile_desc_"+name+".csv")
    df_decile_FF = pf.FF_reg(df_return, df_FF)
    df_decile_FF['alpha'] = df_decile_FF['alpha'] * 12
    df_decile_FF.to_csv(r"..\Data\process_store\logistic\decile_desc_FF_"+name+".csv")    
