# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:39:23 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
#load settings
use_cols = pd.read_csv(r"..\Data\Settings\use_col.csv")

#load data
fund_info = pd.read_csv(r"..\Data\raw_data\fund_info.csv", usecols=use_cols['use_cols'])

#select fund stactic chars
fund_info.dropna(axis=1, how='all', inplace=True)

#select time for fund information
MIN_DATE = 19980101
fund_info = fund_info[fund_info['caldt']>=MIN_DATE]

fund_info_drop_natype = fund_info[fund_info['lipper_class'].notnull()]

fund_info_nan_replace = fund_info_drop_natype.replace({-99.0 : np.nan})

def mod_pipeline(df_fund):
    df_fund.dropna(subset=['tna_latest', 'mgmt_fee', 'exp_ratio', 'turn_ratio', 'exp_ratio', 'turn_ratio', 'vau_fund'], inplace=True)
    df_fund['index_fund_flag'].replace({np.nan:'N'}, inplace=True)
    df_fund['open_to_inv'].replace({np.nan:'Y'}, inplace=True)
    df_fund['et_flag'].replace({np.nan:'No'}, inplace=True)
    return df_fund

fund_info_mod = mod_pipeline(fund_info_nan_replace)
fund_info_mod.to_csv(r"..\Data\raw_data\fund_info_filtered.csv", index=False)
