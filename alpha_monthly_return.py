# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 18:51:50 2020

@author: ycbfr

[alpha monthly return]
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#load fund information and monthly returns
fund_class = pd.read_csv(r"..\Data\Settings\fund_class.csv", index_col=0)
df_month = pd.read_csv(r"..\Data\raw_data\monthly_return_filtered.csv")
df_day = pd.read_csv(r"..\Data\raw_data\daily_return.csv")
df_info = pd.read_csv(r"..\Data\raw_data\fund_info_filtered.csv", usecols=['crsp_fundno', 'lipper_class'])

equity_list = df_info[df_info['lipper_class'].isin(fund_class['Equity'])]['crsp_fundno'].unique()
mix_list = df_info[df_info['lipper_class'].isin(fund_class['Mixed'])]['crsp_fundno'].unique()

df_month_equity = df_month[df_month['crsp_fundno'].isin(equity_list)]
df_month_mix = df_month[df_month['crsp_fundno'].isin(mix_list)]
df_day = df_day[df_day['crsp_fundno'].isin(np.union1d(equity_list, mix_list))]

FF_monthly = pd.read_csv(r"..\Data\FF_monthly.csv", skiprows=3, index_col=0)
FF_daily = pd.read_csv(r"..\Data\FF_daily.csv", skiprows=3, index_col=0)

def add_excess(df):
    df['excess_return'] = np.nan
    df['excess_market'] = np.nan    
    for date in df['caldt'].unique():
        r_fund = df[df['caldt']==date]['mret']
        month = int(date/100)
        r_f = FF_monthly.loc[month, 'RF'] / 100
        r_m = r_fund.median()        
        df.loc[r_fund.index, 'excess_return'] = r_fund - r_f
        df.loc[r_fund.index, 'excess_market'] = r_fund - r_m
    return df

df_month_equity = add_excess(df_month_equity)
df_month_mix = add_excess(df_month_mix)
overlap_list = np.intersect1d(equity_list, mix_list)

df_month = pd.concat([df_month_equity, df_month_mix], axis=0)
df_month = df_month[~df_month['crsp_fundno'].isin(overlap_list)]
df_month.reset_index(drop=True, inplace=True)

#add excess return var
df_month['vol'] = np.nan
df_month['alpha'] = np.nan
df_month['r2'] = np.nan
df_month['excess_return_f'] = np.nan
df_month['excess_market_f'] = np.nan

BETAS_EQUITY = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
for beta in BETAS_EQUITY:
    df_month[beta] = np.nan

for i in range(len(df_month['crsp_fundno'].unique())):
    if i % 76 == 0:
        print(i // 76)
    fundno = df_month['crsp_fundno'].unique()[i]
    fund_month = df_month[df_month['crsp_fundno']==fundno]
    df_month.loc[fund_month.index, 'excess_return_f'] = df_month.loc[fund_month.index, 'excess_return'].shift(-1)
    df_month.loc[fund_month.index, 'excess_market_f'] = df_month.loc[fund_month.index, 'excess_market'].shift(-1)

    fund_day = df_day[df_day['crsp_fundno']==fundno]
    for date in fund_month['caldt'].values:
        year = int(date/10000)
        month = int((date-year*10000)/100)
        date_start = (year - 1) * 10000 + (month + 1) * 100
        date_end = year * 10000 + (month + 1) * 100
        fund_return = fund_day[(fund_day['caldt']<date_end)&(fund_day['caldt']>date_start)].dropna(subset=['dret'])
        if fund_return.shape[0] < 100:
            continue
        else:
            FF_x = FF_daily.loc[fund_return['caldt']] / 100
            RF = FF_x['RF']
            FF_x.drop('RF', axis=1, inplace=True)
            fund_y = fund_return['dret'].astype(float) - RF.values
            model = LinearRegression().fit(FF_x, fund_y)
            fund_month_index = fund_month[fund_month['caldt']==date].index
            df_month.loc[fund_month_index, BETAS_EQUITY] = model.coef_
            df_month.loc[fund_month_index, 'alpha'] = model.intercept_
            df_month.loc[fund_month_index, 'r2'] = model.score(FF_x, fund_y)
            df_month.loc[fund_month_index, 'vol'] = fund_return['dret' ].std()

df_month.to_csv(r"..\Data\monthly_return_add.csv", index=False)