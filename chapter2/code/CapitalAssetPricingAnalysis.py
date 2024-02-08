import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
#import fix_yahoo_finance as yahf
import datetime
import logging

import pandas_datareader.data as financeData
#from pandas_datareader.yahoo.headers import DEFAULT_HEADERS
import yfinance as yahoo_fin
import datetime
import requests_cache
from scipy import stats


class CapitalAssetPricingAnalysis:
  
      def __init__(self):
  	      self.name = "CapitalAssetPrice"
  	      self.data = None
  	      self.dataset_index = 0
		  
      def getCapitalAssetPrices(self,marketIndex,stockSymbol,range_start,range_end):
          yahoo_fin.pdr_override()

          stockData = financeData.get_data_yahoo(stockSymbol,range_start,range_end)
          print("stocksymbol",stockSymbol)
          print("data -",stockData)
          
          marketData = financeData.get_data_yahoo(marketIndex,range_start,range_end)
          print("market Index",marketIndex)
          print("market data -",marketData)
           
          return marketData,stockData
          
      def plotAdjacentPrices(self,marketIndex,stockSymbol,marketData,data):
        
          data['Close'].plot(label = stockSymbol, figsize=(10,8))
          marketData['Close'].plot(label = marketIndex)
          plot.legend()
          plot.show()
          
          
          
      def getLR(self,marketIndex,stockSymbol,marketData,data):
          LR = stats.linregress(data['daily_ret'].iloc[1:],marketData['daily_ret'].iloc[1:])
          
          beta,alpha,r_val,p_val,std_err = LR   
          
          print("beta -",beta,"alpha -",alpha)
            
            
      def plotScatter(self,marketIndex,stockSymbol,marketData,data):
          data['daily_ret'] = data['Close'].pct_change(1)
          marketData['daily_ret'] = marketData['Close'].pct_change(1)
          plot.scatter(data['daily_ret'],marketData['daily_ret'])
          plot.show()
		 
		 
		 		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = CapitalAssetPricingAnalysis()
    
    start_date = datetime.datetime(2003, 9, 29)

    end_date = datetime.datetime(2023, 7, 12)
    
    marketIndex = "SPY"
    stockSymbol = "MSFT"
    period=1
    marketData,stockData = priceIndicator.getCapitalAssetPrices(marketIndex,stockSymbol,start_date,end_date)
    print("plotting")
    priceIndicator.plotAdjacentPrices(marketIndex,stockSymbol,marketData,stockData)
    priceIndicator.plotScatter(marketIndex,stockSymbol,marketData,stockData)
    priceIndicator.getLR(marketIndex,stockSymbol,marketData,stockData)

    

if __name__ == '__main__':
    main()























