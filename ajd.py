# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 17:31:42 2020

@author: ycbfr

[adjustment]
"""
import pandas as pd
import numpy as np
#load data
df_month = pd.read_csv(r"..\Data\monthly_return_add.csv", parse_dates=['caldt'])
df_info = pd.read_csv(r"..\Data\raw_data\fund_info_filtered.csv", usecols=['crsp_fundno', 'caldt', 'exp_ratio', 'turn_ratio'], parse_dates=['caldt'])
df_month['exp_ratio'] = np.nan
df_month['turn_ratio'] = np.nan


for i in range(len(df_month['crsp_fundno'].unique())):
    if i % 76 == 0:
        print(i // 76)
    fundno = df_month['crsp_fundno'].unique()[i]
    fund_month = df_month[df_month['crsp_fundno']==fundno]
    fund_info = df_info[df_info['crsp_fundno']==fundno]
    for date in fund_month['caldt']:
        past_dates = fund_info['caldt'][fund_info['caldt'] < date]
        if len(past_dates) == 0:
            continue
        else:
            closet_date = past_dates.max()
        fund_month_index = fund_month[fund_month['caldt']==date].index
        fund_info_index = fund_info[fund_info['caldt']==closet_date].index
        df_month.loc[fund_month_index, ['exp_ratio', 'turn_ratio']] = df_info.loc[fund_info_index, ['exp_ratio', 'turn_ratio']].values           
        
df_month.to_csv(r"..\Data\monthly_return_final.csv", index=False)        
    
    

