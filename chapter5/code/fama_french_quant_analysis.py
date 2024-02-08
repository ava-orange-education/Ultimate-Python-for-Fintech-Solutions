import numpy as np
import datetime as dtime
import matplotlib.pyplot as matplot
import seaborn as sns
import pandas as pand
from math import sqrt
import os
from scipy import stats


from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
import getFamaFrenchFactors as gff
from pandas_datareader import data as pand_rdr
import matplotlib.colors as mcolors
import math
import statsmodels.api as sm
from statsmodels.api import OLS

import yfinance as yfin


def get_returns(x):
    return (x - x.shift(1))/x.shift(1)
    
def get_stock_data(stock,start_date,end_date):
    yfin.pdr_override()
    prices = []

    stock_dataset = pand.DataFrame()

    stock_dataset[stock] = pand_rdr.get_data_yahoo(stock,
                                        start=start_date,
                                        end = end_date)['Adj Close']
                                        
    print('The number of missing values is: ',stock_dataset.isnull().sum())
    
    
    return stock_dataset 

def plot_logarithmic_prices(stock_dataset,stock):
    matplot.style.use('seaborn')
    matplot.rcParams['figure.figsize'] = [14, 10]
    np.log(stock_dataset).plot()
    matplot.legend(loc='best')
    matplot.title('Logarithmic price',fontsize=18)
    matplot.show()
    
def plot_historical_returns(stock_dataset,returns):
    fig1, axs = matplot.subplots(1, 2,figsize=(20, 10))
    axs[0].hist(returns,bins=round(sqrt(len(returns))))
    axs[1].plot(returns)
    fig1.suptitle('Plots of historical returns',fontsize=18)
    matplot.show()
    
def plot_fama_french_factors(stock_dataset):
    fig3, axs = matplot.subplots(1, 3,figsize=(20, 10))
    axs[0].plot(stock_dataset['Mkt-RF'].rolling(3).mean(),linewidth=2.5)
    axs[0].plot(stock_dataset[stock])
    axs[0].set_title('Stock compared to market returns')
    axs[1].plot(stock_dataset['SMB'].rolling(3).mean(),linewidth=2.5)
    axs[1].plot(stock_dataset[stock])
    axs[1].set_title('Stock compared to small comp returns')
    axs[2].plot(stock_dataset['HML'].rolling(3).mean(),linewidth=2.5)
    axs[2].plot(stock_dataset[stock])
    axs[2].set_title('Stock compared to value stocks index')
    fig3.suptitle('Factors plot',fontsize=18)   
    matplot.show()
    
def executeOLSRegression(stock_dataset,stock,factors):
    Y = stock_dataset[stock] - stock_dataset['RF'] 
    X = stock_dataset[factors]
    model = sm.OLS(Y, sm.tools.add_constant(X))
    result = model.fit()
    print("OLS Regression summary",result.summary())
    return result
     
def execute_fama_french_quant_analysis(stock_dataset,stock,start_date,end_date):
  
    plot_logarithmic_prices(stock_dataset,stock)
    returns = stock_dataset.apply(get_returns)

    plot_historical_returns(stock_dataset,returns)

    mon = stock_dataset.resample('1M').last()
    mon_rets = mon.pct_change().dropna()

    plot_historical_returns(mon_rets,returns)
    

    adf = adfuller(mon_rets)
    print('ADF Statistic: %f' % adf[0])
    print('p-value: %f' % adf[1])


    ff3 = pand.DataFrame(gff.famaFrench3Factor(frequency='m'))

    date_seq = pand.date_range(start=start_date, end=end_date,freq='M')

    ff3 = ff3.rename(columns = {'date_ff_factors':'Date'})

    data = pand.merge(mon_rets,ff3,on='Date',how='left')
    data = data.set_index('Date')

    print('Missing values in the data: ',data.isna().sum())


    data = data.dropna()

    factors = ['Mkt-RF', 'SMB', 'HML']

    plot_fama_french_factors(data)

    cor = data.corr()

    print(f'{stock} correlation - market index:',cor['Mkt-RF'][0])
    print(f'{stock} correlation - small-company portfolio index:',cor['SMB'][0])
    print(f'{stock} correlation - value stocks index:',cor['HML'][0])


    result = executeOLSRegression(data,stock,factors)

    avr = ff3.drop('Date',axis=1).apply(np.mean)

    Int,Mkt,SMB,HML = result.params

    exp_returns = Mkt*avr['Mkt-RF'] + SMB*avr['SMB'] + HML*avr['HML']  

    e_rets = exp_returns - data['RF'].mean()

    print(f'The expected monthly return for {stock} is:',e_rets)
    print(f'The expected anual return for {stock} is:',((1 + e_rets) ** 12) - 1) 


 



if __name__ == "__main__":
  
   start_date = dtime.datetime(2023,4,1)
   end_date = dtime.datetime.now()
   
   stock = 'MSFT'
  
   stock_data = get_stock_data(stock,start_date,end_date)
   
   execute_fama_french_quant_analysis(stock_data,stock,start_date,end_date)