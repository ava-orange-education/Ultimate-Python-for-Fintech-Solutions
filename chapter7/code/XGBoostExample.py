import matplotlib.pyplot as plot
import numpy as nump
import pandas as pand
import fix_yahoo_finance as yahf
from datetime import datetime as dtf
import datetime
import logging
from ta.utils import dropna
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout


from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV

import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics as stock_metrics

import warnings
warnings.filterwarnings('ignore')

class XGBoostExample:
  
      def __init__(self):
  	      self.name = "XGBoostExample"
  	      self.data = None
  	      self.dataset_index = 0
          
      def getStockPrices(self,stockSymbol,range_start,range_end):
    
          print("getting stock price")
          stock_data=yahf.download(stockSymbol,start=range_start,end=range_end)
      
      
          print("data",stock_data.columns)
      
          return stock_data
          
      def plotClosePrices(self,stock_data):
          plot.figure(figsize=(15,5))
          plot.plot(stock_data['Close'])
          plot.title('Stock Close price.', fontsize=15)
          plot.ylabel('Price in dollars.')
          plot.show()
          
      def plotFeatures(self,stock_data):
          stock_data.drop(['Adj Close'], axis=1)
          
          features = ['Open', 'High', 'Low', 'Close', 'Volume']

          plot.subplots(figsize=(20,10))

          for i, col in enumerate(features):
              plot.subplot(2,3,i+1)
              sb.distplot(stock_data[col])
              plot.show()
          
          
          plot.subplots(figsize=(20,10))
          for i, col in enumerate(features):
              plot.subplot(2,3,i+1)
              sb.boxplot(stock_data[col])
              plot.show()
          
          
      def analyzeModel(self,stock_data):
          stock_data.drop(['Adj Close'], axis=1)
          
          print("data", stock_data.head(10))
          days = []
          months = []
          years = []
          for stock_data_date in stock_data.index:
              stock_data_day = stock_data_date.day
              stock_data_month = stock_data_date.month
              stock_data_year = stock_data_date.year   
              
              days.append(stock_data_day)
              months.append(stock_data_month)
              years.append(stock_data_year) 
              
              
          stock_data['day'] = days
          stock_data['month'] = months
          stock_data['year']  = years 
          


          stock_data['is_quarter_end'] = nump.where(stock_data['month']%3==0,1,0)


          data_grouped = stock_data.groupby('year').mean()
          
          stock_data['open-close'] = stock_data['Open'] - stock_data['Close']
          stock_data['low-high'] = stock_data['Low'] - stock_data['High']
          stock_data['target'] = nump.where(stock_data['Close'].shift(-1) > stock_data['Close'], 1, 0)
          
          
          
          stock_features = stock_data[['open-close', 'low-high', 'is_quarter_end']]
          stock_target = stock_data['target']

          stock_scaler = StandardScaler()
          stock_features = stock_scaler.fit_transform(stock_features)

          X_train, X_valid, Y_train, Y_valid = train_test_split(
          	stock_features, stock_target, test_size=0.1, random_state=2022)
          print(X_train.shape, X_valid.shape)

          stock_models = [LogisticRegression(), SVC(
          kernel='poly', probability=True), XGBClassifier()]

          for i in range(3):
              stock_models[i].fit(X_train, Y_train)

              print(f'{stock_models[i]} : ')
              print('Training Accuracy : ', stock_metrics.roc_auc_score(
              	Y_train, stock_models[i].predict_proba(X_train)[:,1]))
              print('Validation Accuracy : ', stock_metrics.roc_auc_score(
              	Y_valid, stock_models[i].predict_proba(X_valid)[:,1]))

          
		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = XGBoostExample()
    
    start_date = datetime.date(2021,6,21)
    end_date = datetime.date(2023,10,20)
    stock_symbol = "MSFT"
    period=1
    training_data = priceIndicator.getStockPrices(stock_symbol,start_date,end_date)
    print("stock data",training_data)

    priceIndicator.plotClosePrices(training_data)
    
    priceIndicator.plotFeatures(training_data)
    
    priceIndicator.analyzeModel(training_data)
    
    


if __name__ == '__main__':
    main()