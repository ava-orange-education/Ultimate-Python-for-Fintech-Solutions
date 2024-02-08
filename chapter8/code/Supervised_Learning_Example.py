import matplotlib.pyplot as plot
import numpy as nump
import pandas as pand
#import fix_yahoo_finance as yahf
import yfinance as yahf
import datetime
import logging


from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class SupervisedLearningExample:
      
      def __init__(self):
  	      self.name = "SupervisedLearningExample"
      
      def getStockPrices(self,stockSymbol,range_start,range_end):
    
          print("getting stock price")
          stock_data=yahf.download(stockSymbol,start=range_start,end=range_end)
      
      
          print("data",stock_data.columns)
      
          return stock_data
      def preProcessData(self,training_data):
          training_data.tail()
          training_data['Open-Close'] = (training_data.Open - training_data.Close)/training_data.Open
          training_data['High-Low'] = (training_data.High - training_data.Low)/training_data.Low
          training_data['percent_change'] = training_data['Adj Close'].pct_change()
          training_data['std_5'] = training_data['percent_change'].rolling(5).std()
          training_data['ret_5'] = training_data['percent_change'].rolling(5).mean()
          training_data.dropna(inplace=True)
          
          return training_data
      def analyzeSupervisedLearningModel(self,X_trained,Y_trained,X_testing,Y_testing,trained_data,train_x):
          rfc = RandomForestClassifier(n_estimators = 100)
          trained_model = rfc.fit(X_trained, Y_trained)

          print('Correct Prediction (%): ', accuracy_score(Y_testing, trained_model.predict(X_testing), normalize=True)*100.0)
          output_report = classification_report(Y_testing, trained_model.predict(X_testing))
          print("output report",output_report)

          trained_data['strategy_returns'] = trained_data.percent_change.shift(-1) * trained_model.predict(train_x)
          
          return trained_data

      def plotStrategyReturns(self,trained_data,split):
          trained_data.strategy_returns[split:].hist()
          plot.xlabel('Strategy returns (%)')
          plot.show()
          
          (trained_data.strategy_returns[split:]+1).cumprod().plot()
          plot.ylabel('Strategy returns (%)')
          plot.show()

      
		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = SupervisedLearningExample()
    
    start_date = datetime.date(2021,6,21)
    end_date = datetime.date(2022,12,25)
    stock_symbol = "MSFT"
    period=1
    training_data = priceIndicator.getStockPrices(stock_symbol,start_date,end_date)
    print("stock data",training_data)

    trained_data = priceIndicator.preProcessData(training_data)
    
    train_x = trained_data[['Open-Close', 'High-Low', 'std_5', 'ret_5']]

    train_y = nump.where(trained_data['Adj Close'].shift(-1) > trained_data['Adj Close'], 1, -1)

    train_dataset_length = trained_data.shape[0]

    split = int(train_dataset_length * 0.75)
    print("split data length is:",split)


    X_trained, X_testing = train_x[:split], train_x[split:]
    Y_trained, Y_testing = train_y[:split], train_y[split:]

    print(X_trained.shape, X_testing.shape)
    print(Y_trained.shape, Y_testing.shape)

    trained_data = priceIndicator.analyzeSupervisedLearningModel(X_trained,Y_trained,X_testing,Y_testing,trained_data,train_x)
    
    priceIndicator.plotStrategyReturns(trained_data,split)


    


if __name__ == '__main__':
    main()