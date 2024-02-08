import matplotlib.pyplot as plot
import numpy as np
import pandas as pand
#import fix_yahoo_finance as yahf
import datetime
import logging

import pandas_datareader.data as financeData
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
              #print("stock data ", stockData)
              #print("stocksymbol",stockSymbol)
              #print("data -",stockData)
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
          
      def getDataFrame(self,stockSymbols,data,allocation):
        
          stock_data = {}
        
          #i=0
          #for stock in stockSymbols:
         
          
          adj_closes = []
          pos_vals = []
          norm_returns = []
          
          dates = []
          
          names = []
          
          allocs = []
          
          i = 0
          for element in data:
              adj_closes = adj_closes + element['Adj Close'].values.tolist()
              element['Normed Return'] = element['Adj Close'] /element.iloc[0]['Adj Close']
              norm_returns = norm_returns + element['Normed Return'].values.tolist()
              dates = dates + element.index.values.tolist()
              
              for date in element.index.values.tolist():
                  names.append(stockSymbols[i])
                  allocs.append(allocation[i])
                  
              i  = i + 1
              #pos_val.append(element['Position Value'])
              
          
          stock_data["Adj Close"] = adj_closes
          #stock_data['Position Value'] = pos_val
          stock_data['Normed Return'] =  norm_returns
          stock_data["Alloc"]  = allocs
          stock_data["Date"] = dates
          stock_data["Name"] = names
 
          #print(" ength Adj close ",len(names), " dates ",len(norm_returns))
          stock_data_frame = pand.DataFrame.from_dict(stock_data)
          return stock_data_frame	 
          
      def analyze_portfolio(self,stock_data_frame):
          #print("stock data frame : ",stock_data_frame)
          stock_data_frame['Allocation'] = stock_data_frame["Normed Return"]*stock_data_frame["Alloc"]
          stock_data_frame["Position Value"] = stock_data_frame['Allocation']*10

          stock_data_frame = stock_data_frame.groupby('Date').agg(Total=pand.NamedAgg(column="Position Value", aggfunc="sum"))
          

          stock_data_frame['Daily Return'] = stock_data_frame['Total'].pct_change(1)

          
          mean = stock_data_frame['Daily Return'].mean()

          print("average daily return is :", mean)
        
          std_dev = stock_data_frame['Daily Return'].std()
          
          print("standard deviation is:",std_dev)

         
          #stock_data_frame['Daily Return'].plot(kind='hist', bins=50, figsize=(4,5))

          


          sharpe_ratio = stock_data_frame['Daily Return'].mean() / stock_data_frame['Daily Return'].std()
          
          print("sharpe ratio is:", sharpe_ratio)

          ASR = (252**0.5) * sharpe_ratio
          
          print("ASR is:", ASR)
          
		 		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = PortfolioAnalysis()
    
    start_date = datetime.datetime(2020, 9, 29)

    end_date = datetime.datetime(2023, 7, 12)
    
    #start_date = "2003-10-29"
    #end_date = "2021-10-11"
    stockSymbols = ['IBM','MSFT','AAPL' , 'GE','AMZN']
    
    #stockSymbols = ['MSFT', 'AMZN', 'AAPL', 'IBM']
    allocation = [.2,.3,.3,.1,.1]
    period=1
    data = priceIndicator.getPortfolioPrices(stockSymbols,start_date,end_date)
    #print("plotting")
    #priceIndicator.plotAdjacentPrices(stockSymbols,data)
    stock_df = priceIndicator.getDataFrame(stockSymbols,data,allocation)
    
    priceIndicator.analyze_portfolio(stock_df)
    
    

    

if __name__ == '__main__':
    main()


