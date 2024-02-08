import matplotlib.pyplot as matplot
import numpy as nump
import pandas as pand
import datetime
import logging

import pandas_datareader as pand_datrdr
import matplotlib.pyplot as plt
import yfinance as yfin
import statsmodels.tools


class FamaFrenchFourFactor:
  
      def __init__(self):
  	      self.name = "FamaFrenchFourFactor"
  	      self.data = None
  	      self.dataset_index = 0
          
      def getStockData(self,stock,start_date):

          yfin.pdr_override()
          

          stock_data = pand_datrdr.data.get_data_yahoo(stock,start=start_date)['Adj Close'].resample('M').ffill().pct_change()
          stock_df = stock_data.to_frame()
          stock_df.index.dtype
          
          stock_df['str_date'] = stock_df.index.astype(str)
          stock_df['dt_date'] = pand.to_datetime(stock_df['str_date']).dt.strftime('%Y-%m')

          stock_df.dt_date.dtype
          
          return stock_df
          
      def getMergedFFData(self,start_date):
          pand_datrdr.famafrench.get_available_datasets()
          data_ff = pand_datrdr.famafrench.FamaFrenchReader('F-F_Research_Data_Factors',freq='M', start=start_date).read()

          data_ff_df = data_ff[0]
          data_ff_df.plot(subplots=True,figsize=(12,4))
          data_ff_df.rolling(72).mean().plot(subplots=True,figsize=(12,4))

          data_ff_mom_df = pand_datrdr.famafrench.FamaFrenchReader('F-F_Momentum_Factor', freq='M',start=start_date).read()[0]

          data_ff_mom_df.rolling(72).mean().plot(subplots=True,figsize=(12,4))

          data_ffac_merged_df = pand.merge(data_ff_df,data_ff_mom_df,on='Date',how='inner',sort=True,copy=True,indicator=False, validate='one_to_one')
          
          return data_ffac_merged_df
          
          
      def executeFamaFrenchAnalysis(self,ffac_merged_df,stock_dfstock_symbol):
          ffac_merged_df['str_date'] = ffac_merged_df.index.astype(str)
          ffac_merged_df['dt_date'] = pand.to_datetime(ffac_merged_df['str_date']).dt.strftime('%Y-%m')


          stock_ffac_merge_df = pand.merge(stock_df,ffac_merged_df,how='inner',on='dt_date',sort=True,copy=True,indicator=False,validate='one_to_one')
          stock_ffac_merge_df

          stock_ffac_merge_df.drop(columns=['str_date_x','str_date_y'],inplace=True)

          stock_ffac_merge_df.rename(columns={'Adj Close':stock_symbol},inplace=True)
          stock_ffac_merge_df['stock_RF'] = stock_ffac_merge_df[stock_symbol]*100-stock_ffac_merge_df['RF']

          stock_ffac_merge_df.dropna(axis=0,inplace=True)

          list(stock_ffac_merge_df)
          stock_ffac_merge_df.rename(columns={'Mom   ':'MOM'}, inplace=True)
          from statsmodels.api import OLS
          results = OLS(stock_ffac_merge_df['stock_RF'],stock_ffac_merge_df[['Mkt-RF','SMB','HML','MOM']],missing='drop').fit()
          print("results summary : ", results.summary())

          stock_ffac_merge_df_c = statsmodels.tools.add_constant(stock_ffac_merge_df,prepend=True)
          results = OLS(stock_ffac_merge_df_c['stock_RF'],stock_ffac_merge_df_c[['const','Mkt-RF','SMB','HML','MOM']],missing='drop').fit()
          print("results summary first element :",results.summary(0))
          
		  
		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    modelAnalyzer = FamaFrenchFourFactor()
    
    start_date = datetime.date(2021,6,21)
    end_date = datetime.date(2023,8,9)
    stock_symbol = "MSFT"
    
    stock_data = modelAnalyzer.getStockData(stock_symbol,start_date)
    
    merged_data_frame_ff = modelAnalyzer.getMergedFFData(start_date)
    
    print(merged_data_frame.head(10))
    
    modelAnalyzer.executeFamaFrenchAnalysis(merged_data_frame_ff,stock_data,stock_symbol)
    
if __name__ == '__main__':
   main()    
    
    
    
    
    
    
   







