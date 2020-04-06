# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:15:30 2020

@author: Chengbo Yang

[compare]
"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

# stats.ttest_rel(rvs1,rvs3)

INPUT_PATH = '../Data/process_store/'
OUTPUT_PATH = '../Data/process_store/compare/'

test = pd.read_csv(INPUT_PATH + 'linear/port_conf_equity.csv', index_col=0)

with pd.ExcelWriter(OUTPUT_PATH + 'T_test.xlsx') as writer:      
    for port in ['equity', 'mix']:    
        df_ttest = pd.DataFrame(index=range(1, 11), columns=['linear', 'logistic', 'LSTM'])
        for mod in ['linear', 'logistic', 'LSTM']:
            input_path_mod = INPUT_PATH + mod + '/'      
            df_return = pd.read_csv(input_path_mod + 'port_return_' + port + '.csv', index_col=0)
            df_return = df_return.iloc[1:]
            for i in range(1,11):
                df_ttest.loc[i, mod] = stats.ttest_rel(df_return[str(i)], df_return['10'])[1]
        df_ttest.to_excel(writer, sheet_name=port)

def add_measure(df_conf):
    df_conf['Accuracy'] = (df_conf['TP'] + df_conf['TN']) / df_conf[['TP', 'FP', 'FN', 'TN']].sum(axis=1)
    df_conf['Precision'] = df_conf['TP'] / (df_conf['TP'] + df_conf['FP'])
    df_conf['Recall'] = df_conf['TP'] / (df_conf['TP'] + df_conf['FN'])
    df_conf['F1_Score'] = 2 * df_conf['Precision'] * df_conf['Recall'] / (df_conf['Precision'] + df_conf['Recall'] )
    return df_conf

for port in ['equity', 'mix']:
    with pd.ExcelWriter(OUTPUT_PATH + 'MC_' + port + '.xlsx') as writer:      
        for measure in ['Accuracy', 'Precision', 'Recall', 'F1_Score']:
            df_measure = pd.DataFrame(index=test.index, columns=['linear', 'logistic', 'LSTM'])
            for mod in ['linear', 'logistic', 'LSTM']:
                input_path_mod = INPUT_PATH + mod + '/'      
                df_conf = pd.read_csv(input_path_mod + 'port_conf_' + port + '.csv', index_col=0)
                df_conf = add_measure(df_conf)
                df_measure[mod] = df_conf[measure].values
            df_measure.iloc[1:].to_excel(writer, sheet_name=measure)

M_LIST = ['Accuracy', 'Precision', 'Recall', 'F1_Score']

for port in ['equity', 'mix']:
    fig, ax = plt.subplots(4, 1, figsize=(20, 60))
    for i in range(len(M_LIST)):
        measure = M_LIST[i]
        df_measure = pd.read_excel(OUTPUT_PATH+ 'MC_' + port + '.xlsx', sheet_name=measure, index_col=0)
        df_measure.plot(figsize=(20, 12.5), ax=ax[i], title=measure)
        fig.tight_layout()
        fig.savefig(OUTPUT_PATH+ 'FC_' + port + '.png')        
                    
for port in ['equity', 'mix']:
    with pd.ExcelWriter(OUTPUT_PATH + 'MSE' + '.xlsx') as writer:      
            df_mean_mse = pd.DataFrame(index=test.index, columns=['linear', 'LSTM'])
            for mod in ['linear', 'LSTM']:
                input_path_mod = INPUT_PATH + mod + '/'
                df_mse = pd.read_csv(input_path_mod + 'mse_' + port + '.csv', index_col=0)
                df_mse['fund_num'] = df_mse['fund_num'] / df_mse['fund_num'].mean()
                df_mse['MSE'] = df_mse['fund_num'] * df_mse['MSE']
                df_mean_mse[mod] = df_mse['MSE'].values
            df_mean_mse.to_excel(writer, sheet_name=port)
            df_mean_mse.plot(figsize=(20,10))
            plt.savefig(OUTPUT_PATH+ 'mse_' + port + '.png')

test_eval = pd.read_csv(INPUT_PATH + 'linear/port_eval_equity.csv', index_col=0)

for port in ['equity', 'mix']:
    with pd.ExcelWriter(OUTPUT_PATH + 'comp_eval_' + port + '.xlsx') as writer:
            for cate in ['high', 'low']:
                df_comp_eval = pd.DataFrame(index=test_eval.columns, columns=['linear', 'logistic', 'LSTM'])
                for mod in ['linear', 'logistic', 'LSTM']:
                    input_path_mod = INPUT_PATH + mod + '/'
                    df_eval = pd.read_csv(input_path_mod + 'port_eval_' + port + '.csv', index_col=0)
                    if cate == 'high':
                        df_comp_eval[mod] = df_eval.loc[1].values
                    elif cate == 'low':
                        df_comp_eval[mod] = df_eval.loc[10].values                 
                df_comp_eval.to_excel(writer, sheet_name=cate)  

FACTOR_LIST = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
def FF_reg(port_return, df_FF):
    X_reg = df_FF[FACTOR_LIST] / 100
    X_reg = sm.add_constant(X_reg)
    y_reg = port_return
    y_reg = y_reg.astype(float)
    fit = sm.OLS(y_reg.values, X_reg.values).fit()    
    return list(fit.params) + list(fit.pvalues)   

DATE_START = pd.Timestamp(2003, 12, 31)
DATE_END = pd.Timestamp(2019, 8, 31)

FF_monthly = pd.read_csv(r"..\Data\FF_monthly.csv", skiprows=3, index_col=0)
FF_monthly.rename(columns={'Mkt-RF':'MKT'}, inplace=True)
df_FF = FF_monthly.loc[DATE_START.year*100+DATE_START.month+1:DATE_END.year*100+DATE_END.month]

test_FF = pd.read_csv(INPUT_PATH + 'linear/decile_desc_FF_equity.csv', index_col=0)            
for port in ['equity', 'mix']:
    with pd.ExcelWriter(OUTPUT_PATH + 'comp_FF_' + port + '.xlsx', engine="openpyxl", mode='a') as writer:
            for cate in ['high', 'low', 'high-low']:
                df_comp_FF = pd.DataFrame(index=test_FF.columns, columns=['linear', 'logistic', 'LSTM'])
                for mod in ['linear', 'logistic', 'LSTM']:
                    input_path_mod = INPUT_PATH + mod + '/'
                    df_return= pd.read_csv(input_path_mod + 'port_return_' + port + '.csv', index_col=0)
                    if cate == 'high':
                        port_return = df_return['1'].iloc[1:]
                    elif cate == 'low':
                        port_return = df_return['10'].iloc[1:]
                    else:
                        port_return = df_return['1'].iloc[1:] - df_return['10'].iloc[1:]
                    df_comp_FF[mod] = FF_reg(port_return, df_FF)
                df_comp_FF.loc['alpha', :] = df_comp_FF.loc['alpha', :] * 12                     
                df_comp_FF.to_excel(writer, sheet_name=cate)  
