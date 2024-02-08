import matplotlib.pyplot as plot
import numpy as nump
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

from scipy.optimize import newton


class BondsAnalysis:
  
      def __init__(self):
  	      self.name = "Bonds"
  	      self.data = None
  	      self.dataset_index = 0
		  

          
      def getYieldToMaturity(self,price,payment_periods,face_value,coupon):
          
          return newton(lambda y: self.getNPVCashFlowValue(y, coupon, face_value, payment_periods) - price, 0.05)
          
      def getNPVCashFlowValue(self,discount, coupon_price, face_value, nperiods):
        
          coeffs = nump.fromfunction(lambda i: 1/(1 + discount)**(i + 1), (nperiods,), dtype=int)

          return self.getWeightedCashFlows(coeffs, coupon_price,face_value,nperiods)
		 
      def getWeightedCashFlows(self,coeffs, coupon_price, face_value, maturity_period):
        
          frequency = 1
          
          future_cash_flows = nump.array([(coupon_price/frequency)*face_value] * (frequency*maturity_period))
          future_cash_flows[-1] += face_value 

          return nump.dot(future_cash_flows, coeffs)
        
         
		  
		 		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceAnalysis = BondsAnalysis()
    
    bond_price = 12
    payment_periods = 8
    face_value = 10
    coupon_age = 0.06
    newton_ytm = priceAnalysis.getYieldToMaturity(bond_price,payment_periods,face_value,coupon_age)
    
    print("yield to Maturity",newton_ytm)
    
    

    

if __name__ == '__main__':
    main()























