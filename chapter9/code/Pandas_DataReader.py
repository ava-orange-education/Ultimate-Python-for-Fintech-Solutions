import pandas as pand
import datetime
import logging

import pandas_datareader as pand_datrdr
import pandas_datareader.famafrench as source
import yfinance as yfin

class PandaDataReader:
  
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
		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)

    modelAnalyzer = PandaDataReader()
    
    start_date = datetime.date(2021,6,21)
    end_date = datetime.date(2023,8,9)
    stock_symbol = "MSFT"
    
    stock_data = modelAnalyzer.getStockData(stock_symbol,start_date)
    
    print("stock data",stock_data)
    
if __name__ == '__main__':
   main()    
    