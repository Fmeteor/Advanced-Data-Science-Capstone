# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:05:21 2019

@author: PJ86YN
"""
import pandas as pd
import numpy as np
#load data
monthly_return = pd.read_csv(r"..\Data\raw_data\monthly_return.csv")
MIN_DATE = 19980901
monthly_return = monthly_return[monthly_return['caldt']>=MIN_DATE]

#replace non-numeric values
monthly_return['mtna'].replace({"T":np.nan}, inplace=True)
monthly_return['mret'].replace({"R":np.nan}, inplace=True)
monthly_return['mnav'].replace({"N":np.nan}, inplace=True)

#convert to float types
monthly_return['mtna'] = monthly_return['mtna'].astype(float)
monthly_return['mret'] = monthly_return['mret'].astype(float)
monthly_return['mnav'] = monthly_return['mnav'].astype(float)

monthly_return = monthly_return.replace({-99.0 : np.nan})
#store the filtered data
monthly_return.to_csv(r"..\Data\raw_data\monthly_return_filtered.csv", index=False)
