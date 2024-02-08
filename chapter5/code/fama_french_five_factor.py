import matplotlib.pyplot as matplot
import numpy as nump
import pandas as pand
import datetime
import logging

import pandas_datareader as pand_datrdr
import pandas_datareader.famafrench as source
import matplotlib.pyplot as plt
import yfinance as yfin
import statsmodels.tools
import statsmodels.formula.api as statsm
from statsmodels.iolib.summary2 import summary_col


class FamaFrenchFiveFactor:
  
      def __init__(self):
  	      self.name = "FamaFrenchFiveFactor"
  	      self.data = None
  	      self.dataset_index = 0
          
      def getStockData(self,stock,start_date):

          yfin.pdr_override()
          

          stock_data = pand_datrdr.data.get_data_yahoo(stock,start=start_date)['Adj Close'].resample('M').ffill().pct_change()
          stock_df = stock_data.to_frame()
          #stock_df.index.dtype
          
          stock_df['str_date'] = stock_df.index.astype(str)
          stock_df['dt_date'] = pand.to_datetime(stock_df['str_date']).dt.strftime('%Y-%m')

          #stock_df.dt_date.dtype
          
          #print("stock data frame :", stock_df.head(10))
          return stock_df
          
      def getMergedFFData(self,start_date):
          
          data_factors = source.FamaFrenchReader('F-F_Research_Data_5_Factors_2x3_daily',freq='M', start=start_date).read()
          data_factors = data_factors[0]
          data_factors.rename(columns={'Mkt-RF': 'MKT'}, inplace=True)
          
          
          #print("data factors :", data_factors.head(10))
          data_factors['MKT'] = data_factors['MKT']/100
          data_factors['SMB'] = data_factors['SMB']/100
          data_factors['HML'] = data_factors['HML']/100
          data_factors['RMW'] = data_factors['RMW']/100
          data_factors['CMA'] = data_factors['CMA']/100
  
          
          
          return data_factors
          
          
      def executeFamaFrenchAnalysis(self,data_stock_factor,stock_df):
        
          data_stock_factor = pand.merge(stock_df,data_stock_factor,left_index=True,right_index=True) 
          #data_stock_factor['XsRet'] = data_stock_factor['Returns'] - data_stock_factor['RF']
          data_stock_factor['XsRet'] = data_stock_factor["Adj Close"]*100-data_stock_factor['RF']
          
          CAPM = statsm.ols(formula = 'XsRet ~ MKT', data=data_stock_factor).fit(cov_type='HAC',cov_kwds={'maxlags':1})
          FF3 = statsm.ols( formula = 'XsRet ~ MKT + SMB + HML', data=data_stock_factor).fit(cov_type='HAC',cov_kwds={'maxlags':1})
          FF5 = statsm.ols( formula = 'XsRet ~ MKT + SMB + HML + RMW + CMA', data=data_stock_factor).fit(cov_type='HAC',cov_kwds={'maxlags':1})

          CAPMtstat = CAPM.tvalues
          FF3tstat = FF3.tvalues
          FF5tstat = FF5.tvalues

          CAPMcoeff = CAPM.params
          FF3coeff = FF3.params
          FF5coeff = FF5.params

          data_results_df = pand.DataFrame({'CAPMcoeff':CAPMcoeff,'CAPMtstat':CAPMtstat,
                                     'FF3coeff':FF3coeff, 'FF3tstat':FF3tstat,
                                     'FF5coeff':FF5coeff, 'FF5tstat':FF5tstat},
          index = ['Intercept', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])


          data_dfoutput = summary_col([CAPM,FF3, FF5],stars=True,float_format='%0.4f',
                        model_names=['CAPM','FF3','FF5'],
                        info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                                   'Adjusted R2':lambda x: "{:.4f}".format(x.rsquared_adj)}, 
                                   regressor_order = ['Intercept', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])

          print("results summary first element :", data_dfoutput)
          
		  
		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)

    modelAnalyzer = FamaFrenchFiveFactor()
    
    start_date = datetime.date(2021,6,21)
    end_date = datetime.date(2023,8,9)
    stock_symbol = "MSFT"
    
    stock_data = modelAnalyzer.getStockData(stock_symbol,start_date)
    
    merged_data_frame_ff = modelAnalyzer.getMergedFFData(start_date)
    
    #print(merged_data_frame.head(10))
    
    modelAnalyzer.executeFamaFrenchAnalysis(merged_data_frame_ff,stock_data)
    
if __name__ == '__main__':
   main()    
    
    
    
    
    
    
   







