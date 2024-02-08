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


class BlackSholesAnalyzer:
  
      def __init__(self):
  	      self.name = "BlackSholesAnalyzer"
  	      self.data = None
  	      self.dataset_index = 0
		  
          

      def execute_call(self,S, K, T, r, sigma,Num):
          dat1 = (nump.log(S/K) + (r + sigma**2/2)*T) / (sigma*nump.sqrt(T))
          dat2 = dat1 - sigma * nump.sqrt(T)
          return S * Num(dat1) - K * nump.exp(-r*T)* Num(dat2)

      def execute_put(self,S, K, T, r, sigma,Num):
          dat1 = (nump.log(S/K) + (r + sigma**2/2)*T) / (sigma*nump.sqrt(T))
          dat2 = dat1 - sigma* nump.sqrt(T)
          return K*nump.exp(-r*T)*Num(-dat2) - S*Num(-dat1)
          
      def analyze_Black_Sholes_Model(self,S,calls,puts):
          plot.plot(S, calls, label='Call Value')
          plot.plot(S, puts, label='Put Value')
          plot.xlabel('$S_0$')
          plot.ylabel(' Value')
          plot.legend()   
          plot.show()   
          
		 		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = BlackSholesAnalyzer()
    
    Num = normalize.cdf
    K = 100
    r = 0.1
    T = 1
    sigma = 0.3
    
    arrS = nump.arange(60,140,0.1)

    stock_calls = [priceIndicator.execute_call(s, K, T, r, sigma,Num) for s in arrS]
    stock_puts = [priceIndicator.execute_put(s, K, T, r, sigma,Num) for s in arrS]
    
    priceIndicator.analyze_Black_Sholes_Model(arrS,stock_calls,stock_puts)
    
    

    

if __name__ == '__main__':
    main()
