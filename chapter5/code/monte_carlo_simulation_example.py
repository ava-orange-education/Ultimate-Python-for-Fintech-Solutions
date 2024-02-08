import matplotlib.pyplot as plot
import numpy as nump
import pandas as pand
import datetime
import logging

import pandas_datareader.data as financeData
import yfinance as yahoo_fin
import datetime
import requests_cache
from scipy.stats import norm as normalize


class MonteCarloSimulation:
  
      def __init__(self):
  	      self.name = "MonteCarloSimulation"
  	      self.data = None
  	      self.dataset_index = 0
		  
      def getPortfolioPrices(self,stockSymbol,range_start,range_end):
          yahoo_fin.pdr_override()

          stock_data = financeData.get_data_yahoo([stockSymbol], 
                           start=range_start, 
                           end=range_end)['Adj Close']
                           
          print("stock data:", stock_data.head(10))
          return stock_data
          
    

          
      def simulate_monte_carlo(self,stock_symbol,stock_data,num_days_to_test,num_days_to_predict,numSimulations):
          daily_return = nump.log(1 + stock_data.pct_change())
          average_daily_return = daily_return.mean()
          variance = daily_return.var()
          drift = average_daily_return - (variance/2)
          standard_deviation = daily_return.std()

          predictions = nump.zeros(num_days_to_test+num_days_to_predict)
          predictions[0] = stock_data[-num_days_to_test]
          pred_collection = nump.ndarray(shape=(numSimulations,num_days_to_test+num_days_to_predict))


          for j in range(0,numSimulations):
              for i in range(1,num_days_to_test+num_days_to_predict):
                  random_value = standard_deviation * normalize.ppf(nump.random.rand())
                  predictions[i] = predictions[i-1] * nump.exp(drift + random_value)
              pred_collection[j] = predictions
	
          differences = nump.array([])
          for k in range(0,numSimulations):
              difference_arrays = nump.subtract(stock_data.values[-30:],pred_collection[k][:-1])
              difference_values = nump.sum(nump.abs(difference_arrays))
              differences = nump.append(differences,difference_values)
    
          best_fit = nump.argmin(differences)
          future_price = pred_collection[best_fit][-1]
          
          print("best fit :", best_fit)
          print("future_price: ",future_price)
          
		 		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = MonteCarloSimulation()
    
    num_days_to_test = 30      
    num_days_to_predict = 1    
    numSimulations = 1000     
    stock_symbol = 'MSFT'        
    
    start_date = datetime.datetime(2020, 9, 29)

    end_date = datetime.datetime(2023, 7, 12)
    
    stock_data = priceIndicator.getPortfolioPrices(stock_symbol,start_date,end_date)    
    priceIndicator.simulate_monte_carlo(stock_symbol,stock_data,num_days_to_test,num_days_to_predict,numSimulations)
    
    

    

if __name__ == '__main__':
    main()
