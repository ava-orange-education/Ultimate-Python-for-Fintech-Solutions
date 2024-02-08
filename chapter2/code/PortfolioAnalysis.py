import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
#import fix_yahoo_finance as yahf
import datetime
import logging

#import pandas_datareader.data as financeData
#from pandas_datareader.yahoo.headers import DEFAULT_HEADERS
import yfinance as yahoo_fin
import datetime
import requests_cache


class PortfolioAnalysis:
  
      def __init__(self):
  	      self.name = "PortfolioAnalysis"
  	      self.data = None
  	      self.dataset_index = 0
		  
      def getPortfolioPrices(self,stockSymbols,range_start,range_end):
          yahoo_fin.pdr_override()
          portfolio_data = []
          for stockSymbol in stockSymbols:
              stockData = financeData.get_data_yahoo(stockSymbol,range_start,range_end)
              print("stocksymbol",stockSymbol)
              print("data -",stockData)
              portfolio_data.append(stockData) 
          return portfolio_data
          
      def plotAdjacentPrices(self,stockSymbols,data):
        
         for i in range(0,len(stockSymbols)):
             stockSymbol = stockSymbols[i]
             stockData = data[i]
             self.plotStockAdjPrices(stockSymbol,stockData)
            
            
      def plotStockAdjPrices(self,stockSymbol,data):
          plot.figure(figsize=(15, 7))
          plotD  =data['Adj Close'].plot()
          #print("data",plotD)
          plot.title(stockSymbol + '-  Data', fontsize=16)
          plot.xlabel('Year', fontsize=15)
          plot.ylabel('Price ($)', fontsize=15)
          plot.xticks(fontsize=15)
          plot.yticks(fontsize=15)
          plot.legend(['Close'], prop={'size': 15})
          plot.show()
		 
		 
		 		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = PortfolioAnalysis()
    
    start_date = datetime.datetime(2003, 9, 29)

    end_date = datetime.datetime(2023, 7, 12)
    
    #start_date = "2003-10-29"
    #end_date = "2021-10-11"
    stockSymbols = ['^SP500TR','MSFT','AAPL' , 'GE','^BCOM']
    period=1
    data = priceIndicator.getPortfolioPrices(stockSymbols,start_date,end_date)
    print("plotting")
    priceIndicator.plotAdjacentPrices(stockSymbols,data)

    

if __name__ == '__main__':
    main()























