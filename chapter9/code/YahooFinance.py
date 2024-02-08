import matplotlib.pyplot as plot
import numpy as np
import pandas as pand
import yfinance as yahf
import datetime
import logging

class YahooFinance:
  
      def __init__(self):
  	      self.name = "TechnicalIndicatorsAnalyzer"
  	      self.data = None
  	      self.dataset_index = 0
      def getStockPrices(self,stockSymbol,range_start,range_end):
    
          print("getting stock price")
          stock_data=yahf.download(stockSymbol,start=range_start,end=range_end)
      
      
          print("data",stock_data.columns)
      
          return stock_data

def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = YahooFinance()
    
    start_date = datetime.date(2021,6,21)
    end_date = datetime.date(2023,8,9)
    stock_symbol = "MSFT"
    period=1
    data = priceIndicator.getStockPrices(stock_symbol,start_date,end_date)
    print("stock data",data)
    


if __name__ == '__main__':
    main()          
