import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import fix_yahoo_finance as yahf
import datetime
import logging


class FuturesAnalysis:
  
      def __init__(self):
  	      self.name = "FuturesPrice"
  	      self.data = None
  	      self.dataset_index = 0
		  
      def getFuturePrices(self,stockSymbol,range_start,range_end):
    
          print("getting futures ")
          
          data = yahf.download(stockSymbol, range_start, range_end)
          data.index = pd.to_datetime(data.index)
          
          print("futures",data.head(50))
          
          return data
          
      def plotAdjacentPrices(self,stockSymbol,data):
        
          plot.figure(figsize=(15, 7))
          data['Adj Close'].plot()
          plot.title(stockSymbol + '- Futures Data', fontsize=16)
          plot.xlabel('Year', fontsize=15)
          plot.ylabel('Price ($)', fontsize=15)
          plot.xticks(fontsize=15)
          plot.yticks(fontsize=15)
          plot.legend(['Close'], prop={'size': 15})
          plot.show()
		 
		 
		 		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = FuturesAnalysis()
    
    start_date = datetime.date(2021,6,21)
    end_date = datetime.date(2023,8,9)
    stock_symbol = "GC=F"
    period=1
    data = priceIndicator.getFuturePrices(stock_symbol,start_date,end_date)
    priceIndicator.plotAdjacentPrices(stock_symbol,data)

    

if __name__ == '__main__':
    main()