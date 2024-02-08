import matplotlib.pyplot as plot
import numpy as nump
import pandas as pand
#import fix_yahoo_finance as yahf
from yahooquery import Ticker
import datetime
import logging
import math


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class TimeSeriesExample:
      
      def __init__(self):
  	      self.name = "TimeSeriesExample"
      
      def getStockPrices(self,stockSymbol,range_start,range_end):
    
          print("getting stock price")
          #stock_data=yahf.download(stockSymbol,start=range_start,end=range_end)

          stock_data = Ticker(stockSymbol)
      
      
          print("data",stock_data)
      
          return stock_data
      
      def plotMovingAverage(self,stock_std_dev,stock_moving_avg):
          plot.legend(loc='best')
          plot.title('Moving Average')
          plot.plot(stock_std_dev, color ="black", label = "Standard Deviation")
          plot.plot(stock_moving_avg, color="red", label = "Mean")
          plot.legend()
          plot.show()
          
      def check_stationarity(self,training_data):
          rolmean = training_data.rolling(12).mean()
          rolstd = training_data.rolling(12).std()
          plot.plot(training_data, color='blue',label='Original')
          plot.plot(rolmean, color='red', label='Rolling Mean')
          plot.plot(rolstd, color='black', label = 'Rolling Std')
          plot.legend(loc='best')
          plot.title('Rolling Mean and Standard Deviation')
          plot.show(block=False)
          print("Dickey fuller test - output")
          adft = adfuller(training_data,autolag='AIC')
          output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
          for key,values in adft[4].items():
              output['critical value (%s)'%key] =  values
          print(output)
      
		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = TimeSeriesExample()
    
    start_date = datetime.date(2021,6,21)
    end_date = datetime.date(2023,10,25)
    stock_symbol = "MSFT"
    period=1
    training_data = priceIndicator.getStockPrices(stock_symbol,start_date,end_date)
    print("stock data",training_data.history())	
    
    history = training_data.history()
    stock_df_close = history["close"]
    stock_df_close.plot(kind='kde')

    priceIndicator.check_stationarity(stock_df_close)
    
    stock_df_log = nump.log(stock_df_close)
    stock_moving_avg = stock_df_log.rolling(12).mean()
    stock_std_dev = stock_df_log.rolling(12).std()	  

    priceIndicator.plotMovingAverage(stock_moving_avg,stock_std_dev)


    
    stock_train_data,stock_test_data = stock_df_log[3:int(len(stock_df_log)*0.9)], stock_df_log[int(len(stock_df_log)*0.9):]
    
    stock_model_autoARIMA = auto_arima(stock_train_data, start_p=0, start_q=0,
                          test='adf',       
                          max_p=3, max_q=3, 
                          m=1,              
                          d=None,         
                          seasonal=False,   
                          start_P=0, 
                          D=0, 
                          trace=True,
                          error_action='ignore',  
                          suppress_warnings=True, 
                          stepwise=True)
    print(model_autoARIMA.summary())
    model_autoARIMA.plot_diagnostics(figsize=(15,8))
    
    stock_model = ARIMA(stock_train_data, order=(1,1,2))  
    stock_fitted = stock_model.fit(disp=-1)  
    print("Model Fit ",stock_fitted.summary())


    stock_fc, stock_se, stock_conf = stock_fitted.forecast(321, alpha=0.05) 
    stock_fc_series = pand.Series(stock_fc, index=stock_test_data.index)
    stock_lower_series = pand.Series(stock_conf[:, 0], index=stock_test_data.index)
    stock_upper_series = pand.Series(cstock_onf[:, 1], index=stock_test_data.index)
    

    stock_mse = mean_squared_error(stock_test_data, stock_fc)
    print('MSE: '+str(stock_mse))
    stock_mae = mean_absolute_error(stock_test_data, stock_fc)
    print('MAE: '+str(stock_mae))
    stock_rmse = math.sqrt(mean_squared_error(stock_test_data, stock_fc))
    print('RMSE: '+str(stock_rmse))
    stock_mape = nump.mean(nump.abs(stock_fc - stock_test_data)/nump.abs(stock_test_data))
    print('MAPE: '+str(stock_mape))
		  
if __name__ == '__main__':
    main()