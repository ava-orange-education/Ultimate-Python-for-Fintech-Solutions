import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
#import fix_yahoo_finance as yahf
#import pandas_datareader.data as yahweb
#from google_finance import *
from yahoo_fin import options,stock_info
import datetime
import logging


class OptionsAnalysis:
  
      def __init__(self):
  	      self.name = "OptionsPrice"
  	      self.data = None
  	      self.dataset_index = 0
		  
      def getOptionPrices(self,stockSymbol,range_start,range_end):
    
          print("getting option calls")
          
          calls = options.get_calls(stockSymbol)
          puts = options.get_puts(stockSymbol)
          
          print("calls",calls)
          
          print("puts",puts)
          
          return calls,puts	 
          
      def getAdjacentMarketPrice(self,stockSymbol,calls,puts):
        
          last_adj_close = stock_info.get_data(stockSymbol)["adjclose"][-1]

          #calls = options.get_calls("aapl")
          #puts = options.get_puts("aapl")

          adjm_call = calls.iloc[(calls["Strike"] - last_adj_close).abs().argsort()[:1]]
          
          adjm_put = puts.iloc[(puts["Strike"] - last_adj_close).abs().argsort()[:1]]
          
          print("adjacent Market price call",adjm_call)
          
          print("adjacent Market Price put",adjm_put)
		 
		 
		 		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = OptionsAnalysis()
    
    start_date = datetime.date(2021,6,21)
    end_date = datetime.date(2023,8,9)
    stock_symbol = "MSFT"
    period=1
    data = priceIndicator.getOptionPrices(stock_symbol,start_date,end_date)
    calls, puts = data
    
    priceIndicator.getAdjacentMarketPrice(stock_symbol,calls,puts)

    

if __name__ == '__main__':
    main()