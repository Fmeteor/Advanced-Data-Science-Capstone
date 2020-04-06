  # -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:55:36 2020

@author: ycbfr

[calculate data description tables]
"""
import pandas as pd
import numpy as np
import project_functions as pf

df = pd.read_csv(r"..\Data\monthly_return_final.csv", parse_dates=['caldt'])
df_info = pd.read_csv(r"..\Data\fund_info_add.csv")
df_class = pd.read_csv(r"..\Data\Settings\fund_class.csv", index_col=0)

df.rename(columns={'caldt':'date', 'crsp_fundno':'fundno'}, inplace=True)

BETAS = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
var_list = ['mtna', 'vol', 'alpha', 'exp_ratio', 'turn_ratio', 'r2'] + BETAS

list_euqity = df_info[df_info['lipper_class'].isin(df_class['Equity'])]['crsp_fundno'].unique()
list_mix = df_info[df_info['lipper_class'].isin(df_class['Mixed'])]['crsp_fundno'].unique()

#group funds by asset class 
df_equity = df[df['fundno'].isin(list_euqity)].dropna(subset=var_list)
df_mix = df[df['fundno'].isin(list_mix)].dropna(subset=var_list)

def data_description(df):
    df['alpha'] = df['alpha'] * 252
    df['vol'] = df['vol'] * np.sqrt(252)
    df['year'] = df['date'].dt.year
    df_desc = df.groupby(['year'])[var_list].median().round(2)
    df_desc['fundnum'] = df.groupby(['year'])['fundno'].nunique()
    YEAR_START = df['year'].min()
    YEAR_END = df['year'].max()
    df_desc.loc[str(YEAR_START) + '-' + str(YEAR_END), var_list] = df[var_list].median().round(2)
    df_desc.loc[str(YEAR_START) + '-' + str(YEAR_END), 'fundnum'] = df['fundno'].nunique()
    df_desc['fundnum'] = df_desc['fundnum'].astype(int)
    df_desc = df_desc[['fundnum']+var_list]
    df_desc.reset_index(inplace=True)
    return df_desc

des_equity = data_description(df_equity)
des_mix = data_description(df_mix)

with pd.ExcelWriter(r'..\Data\process_store\data_description.xlsx') as writer:
    des_equity.to_excel(writer, sheet_name='Equity')
    des_mix.to_excel(writer, sheet_name='Mix')

with open(r"..\Data\process_store\data_desc.txt", 'w') as desc_file:
    desc_file.write(des_equity.to_latex(index=False))
    desc_file.write("\n\n")
    desc_file.write(des_mix.to_latex(index=False))

df_equity.dropna(subset=['excess_market', 'excess_market_f'], inplace=True)
df_mix.dropna(subset=['excess_market', 'excess_market_f'], inplace=True)  
var_list = ['mtna', 'excess_market', 'vol', 'alpha', 'exp_ratio', 'turn_ratio', 'r2'] + BETAS

def add_decile(df):
    df['decile'] = pd.qcut(df['excess_market_f'], 10, labels=range(1,11))
    df['decile'] = 11 - df['decile'].astype(int)
    return df

df_equity = df_equity.groupby('date').apply(add_decile)
df_mix = df_mix.groupby('date').apply(add_decile)


with open(r"..\Data\process_store\decile_desc.txt", 'w') as decile_file:
    for df in [df_equity, df_mix]:
        df_decile = pf.decile_description(df, var_list)
        df_decile['excess_market'] = df_decile['excess_market'] * 12
        decile_file.write(df_decile.round(4).to_latex())
        decile_file.write("\n\n")