import matplotlib.pyplot as plot
import numpy as nump
import pandas as pand
import datetime
import logging
import seaborn as seab

from yahooquery import Ticker






class YahooQuery:
      
      def __init__(self):
  	      self.name = "NeuralNetworkExample"
      
      def getStockPrices(self,stockSymbol):
    
          stock_data = Ticker(stockSymbol)
          #print("columns ",stock_data.columns)
          
          stock_price_history = stock_data.history(period='max', interval='1d')
          
          return stock_price_history
		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = YahooQuery()
    


    stock_symbol = "MSFT"


    #period=1
    stock_data = priceIndicator.getStockPrices(stock_symbol)
    print("stock data",stock_data)		  
		  
if __name__ == '__main__':
    main()