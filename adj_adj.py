# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 10:50:07 2020

@author: ycbfr
"""

import pandas as pd
import numpy as np

df = pd.read_csv(r"..\Data\monthly_return_final.csv")
df = df.replace({-99.0 : np.nan})

df.to_csv(r"..\Data\monthly_return_final.csv", index=False)