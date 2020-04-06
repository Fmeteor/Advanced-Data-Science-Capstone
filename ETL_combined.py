# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:14:44 2019

@author: PJ86YN
"""
import pandas as pd

monthly_return = pd.read_csv(r"..\Data\raw_data\monthly_return_filtered.csv")
fund_info = pd.read_csv(r"..\Data\raw_data\fund_info_filtered.csv")

inter_ids = set(monthly_return['crsp_fundno']) & set(fund_info['crsp_fundno'])

fund_info_inter = fund_info[fund_info['crsp_fundno'].isin(inter_ids)]
monthly_return_inter = monthly_return[monthly_return['crsp_fundno'].isin(inter_ids)]

monthly_return_inter.to_csv(r"..\Data\raw_data\monthly_return_inter.csv", index=False)
fund_info_inter.to_csv(r"..\Data\raw_data\fund_info_inter.csv", index=False)    
