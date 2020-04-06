# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 20:25:13 2020

@author: ycbfr

[ETL_daily_return]
"""
import pandas as pd
import numpy as np
#load data
daily_return = pd.read_csv(r"..\Data\raw_data\daily_return.csv")

#replace non-numeric values
daily_return['dret'].replace({"R":np.nan}, inplace=True)

#convert to float types
daily_return['dret'] = daily_return['dret'].astype(float)


#store the filtered data
daily_return.to_csv(r"..\Data\raw_data\daily_return.csv", index=False)

