# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 09:42:30 2019

@author: PJ86YN
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#load fund information and monthly returns
fund_class = pd.read_csv(r"..\Data\Settings\fund_class.csv", index_col=0)
monthly_return = pd.read_csv(r"..\Data\raw_data\monthly_return_inter.csv")
fund_info = pd.read_csv(r"..\Data\raw_data\fund_info_inter.csv")

#load FF Factors
FF_monthly = pd.read_csv(r"..\Data\FF_monthly.csv", skiprows=3, index_col=0)
FF_yearly = pd.read_csv(r"..\Data\FF_yearly.csv", index_col=0)

#load BOND Factors
BOND_monthly = pd.read_csv(r"..\Data\BOND_monthly.csv", index_col=0)
BOND_yearly = pd.read_csv(r"..\Data\BOND_yearly.csv", index_col=0)

#add excess return var
fund_info['excess_return'] = np.nan
fund_info['excess_return_f'] = np.nan
for date in fund_info['caldt'].unique():
    r_fund =  fund_info[fund_info['caldt']==date]['yret']
    year = int(date/10000)
    r_f = FF_yearly.loc[year, 'RF'] / 100        
    fund_info.loc[r_fund.index, 'excess_return'] = r_fund - r_f

#convert date to year
fund_info['caldt'] = (fund_info['caldt'] / 10000).astype(int)
monthly_return['caldt'] = (monthly_return['caldt'] / 100).astype(int)

BETAS_EQUITY = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
BETAS_BOND = ['DEF', 'TERM']
BETAS_MIX = BETAS_EQUITY + BETAS_BOND
fund_info['alpha'] = np.nan
for beta in BETAS_MIX:
    fund_info[beta] = np.nan

#group funds by asset class 
fund_equity = fund_info[fund_info['lipper_class'].isin(fund_class['Equity'])]
fund_bond = fund_info[fund_info['lipper_class'].isin(fund_class['Fixed_Income'])]
fund_mix = fund_info[fund_info['lipper_class'].isin(fund_class['Mixed'])]

for fundno in fund_equity['crsp_fundno'].unique():
    fund_df = fund_equity[fund_equity['crsp_fundno']==fundno]
    fund_equity.loc[fund_df.index, 'excess_return_f'] = fund_equity.loc[fund_df.index, 'excess_return'].shift(-1)
    fund_mreturn = monthly_return[monthly_return['crsp_fundno']==fundno]
    dates_list = fund_df['caldt'].values
    for year in dates_list:
        fund_return = fund_mreturn[(fund_mreturn['caldt']<(year+1)*100)&(fund_mreturn['caldt']>(year-2)*100)].dropna(subset=['mret'])
        if fund_return.shape[0] < 30:
            continue
        else:
            FF_x = FF_monthly.loc[fund_return['caldt']] / 100
            RF = FF_x['RF']
            FF_x.drop('RF', axis=1, inplace=True)
            fund_y = fund_return['mret'].astype(float) - RF.values
            model = LinearRegression().fit(FF_x, fund_y)
            fund_year_index = fund_df[fund_df['caldt']==year].index
            fund_equity.loc[fund_year_index, BETAS_EQUITY] = model.coef_
            fund_equity.loc[fund_year_index, 'alpha'] = model.intercept_ * 12

for fundno in fund_bond['crsp_fundno'].unique():
    fund_df = fund_bond[fund_bond['crsp_fundno']==fundno]
    fund_bond.loc[fund_df.index, 'excess_return_f'] = fund_bond.loc[fund_df.index, 'excess_return'].shift(-1)
    fund_mreturn = monthly_return[monthly_return['crsp_fundno']==fundno]
    dates_list = fund_df['caldt'].values
    for year in dates_list:
        fund_return = fund_mreturn[(fund_mreturn['caldt']<(year+1)*100)&(fund_mreturn['caldt']>(year-2)*100)].dropna(subset=['mret'])
        if fund_return.shape[0] < 30:
            continue
        else:
            BOND_x = BOND_monthly.loc[fund_return['caldt']]
            RF = BOND_x['RF']
            BOND_x.drop('RF', axis=1, inplace=True)
            fund_y = fund_return['mret'].astype(float) - RF.values
            model = LinearRegression().fit(BOND_x, fund_y)
            fund_year_index = fund_df[fund_df['caldt']==year].index
            fund_bond.loc[fund_year_index, BETAS_BOND] = model.coef_
            fund_bond.loc[fund_year_index, 'alpha'] = model.intercept_ * 12

for fundno in fund_mix['crsp_fundno'].unique():
    fund_df = fund_mix[fund_mix['crsp_fundno']==fundno]
    fund_mix.loc[fund_df.index, 'excess_return_f'] = fund_mix.loc[fund_df.index, 'excess_return'].shift(-1)
    fund_mreturn = monthly_return[monthly_return['crsp_fundno']==fundno]
    dates_list = fund_df['caldt'].values
    for year in dates_list:
        fund_return = fund_mreturn[(fund_mreturn['caldt']<(year+1)*100)&(fund_mreturn['caldt']>(year-2)*100)].dropna(subset=['mret'])
        if fund_return.shape[0] < 30:
            continue
        else:
            BOND_x = BOND_monthly.loc[fund_return['caldt']]
            FF_x = FF_monthly.loc[fund_return['caldt']] / 100
            RF = FF_x['RF']
            FF_x.drop('RF', axis=1, inplace=True)
            BOND_x.drop('RF', axis=1, inplace=True)
            MIX_x = pd.concat([FF_x, BOND_x], axis=1)
            fund_y = fund_return['mret'].astype(float) - RF.values
            model = LinearRegression().fit(MIX_x, fund_y)
            fund_year_index = fund_df[fund_df['caldt']==year].index
            fund_mix.loc[fund_year_index, BETAS_MIX] = model.coef_
            fund_mix.loc[fund_year_index, 'alpha'] = model.intercept_ * 12

fund_info_add = pd.concat([fund_equity, fund_bond, fund_mix], axis=0)
    
fund_info_add.to_csv(r"..\Data\fund_info_add.csv", index=False)