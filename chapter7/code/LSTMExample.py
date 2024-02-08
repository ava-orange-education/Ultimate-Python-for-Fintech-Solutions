import matplotlib.pyplot as plot
import numpy as nump
import pandas as pand
import fix_yahoo_finance as yahf
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

class LSTMExample:
  
      def __init__(self):
  	      self.name = "LSTMExample"
  	      self.data = None
  	      self.dataset_index = 0
          
      def getStockPrices(self,stockSymbol,range_start,range_end):
    
          print("getting stock price")
          stock_data=yahf.download(stockSymbol,start=range_start,end=range_end)
      
      
          print("data",stock_data.columns)
      
          return stock_data
       
      def getmodel_regressionModel(self,optimizer):
    	      model_regression= Sequential()


    	      model_regression.add(LSTM(units=50,return_sequences=True,kernel_initializer='glorot_uniform',input_shape=(xtrain.shape[1],1)))
    	      model_regression.add(Dropout(0.2))


    	      model_regression.add(LSTM(units=50,kernel_initializer= 'glorot_uniform',return_sequences=True))
    	      model_regression.add(Dropout(0.2))


    	      model_regression.add(LSTM(units=50,kernel_initializer='glorot_uniform',return_sequences=True))
    	      model_regression.add(Dropout(0.2))


    	      model_regression.add(LSTM(units=50,kernel_initializer='glorot_uniform'))
    	      model_regression.add(Dropout(0.2))

    	      model_regression.add(Dense(units=1))

    	      model_regression.compile(optimizer=optimizer,loss='mean_squared_error')

    	      return model_regression
     
      def analyzeLSTM(self,training_data,testing_data):

            print("training data",training_data.head())
            market_train_open= training_data.iloc[:, 1:2].values
            ss= MinMaxScaler(feature_range=(0,1))
            market_train_open_scaled= ss.fit_transform(market_train_open)
            market_train_open_scaled[60]
            xtrain=[]
            ytrain=[]
            for i in range(60,len(market_train_open_scaled)):
                xtrain.append(market_train_open_scaled[i-60:i,0])
                ytrain.append(market_train_open_scaled[i,0])
            xtrain, ytrain = nump.array(xtrain), nump.array(ytrain)
            xtrain= nump.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],1))
            print("xtrain shape",xtrain.shape)
            model_regression= Sequential()
            model_regression.add(LSTM(units=50,return_sequences=True,kernel_initializer='glorot_uniform',input_shape=(xtrain.shape[1],1)))
            model_regression.add(Dropout(0.2))
            model_regression.add(LSTM(units=50,kernel_initializer='glorot_uniform',return_sequences=True))
            model_regression.add(Dropout(0.2))
            model_regression.add(LSTM(units=50,kernel_initializer='glorot_uniform',return_sequences=True))
            model_regression.add(Dropout(0.2))
            model_regression.add(LSTM(units=50,kernel_initializer='glorot_uniform'))
            model_regression.add(Dropout(0.2))
            model_regression.add(Dense(units=1))
            model_regression.compile(optimizer='adam',loss='mean_squared_error')
            model_regression.fit(xtrain,ytrain,batch_size=30,epochs=100)
            market_test_open= testing_data.iloc[:, 1:2].values 
            total_data= pand.concat([training_data['Open'],testing_data['Open']],axis=0) 
            market_test_input = total_data[len(total_data)-len(testing_data)-60:].values
            market_test_input= market_test_input.reshape(-1,1)
            market_test_input= ss.transform(market_test_input)


            market_xtest= []
            for i in range(60,80):
                market_xtest.append(market_test_input[i-60:i,0]) 
            market_xtest= nump.array(market_xtest)
            market_xtest= nump.reshape(market_xtest,(market_xtest.shape[0],market_xtest.shape[1],1))
            market_predicted_value= model_regression.predict(market_xtest)
            market_predicted_value= ss.inverse_transform(market_predicted_value)
            
            return market_test_open, market_predicted_value


      def plotPrices(self,market_open,market_prediction):

          plot.figure(figsize=(20,10))
          plot.plot(market_open,'red',label='Real Prices')
          plot.plot(market_prediction,'blue',label='Predicted Prices')
          plot.xlabel('Time')
          plot.ylabel('Prices')
          plot.title('Real vs Predicted Prices')
          plot.legend(loc='best', fontsize=20)
          plot.show()
          



	      
		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = LSTMExample()
    
    start_date = datetime.date(2021,6,21)
    end_date = datetime.date(2022,12,25)
    stock_symbol = "MSFT"
    period=1
    training_data = priceIndicator.getStockPrices(stock_symbol,start_date,end_date)
    print("stock data",training_data)

    start_date = datetime.date(2023,1,21)
    end_date = datetime.date(2023,9,25)
    
    test_data = priceIndicator.getStockPrices(stock_symbol,start_date,end_date)
    
    print("stock test data",test_data)
    
    market_open, market_prediction = priceIndicator.analyzeLSTM(training_data,test_data)
    
    priceIndicator.plotPrices(market_open,market_prediction)
    


if __name__ == '__main__':
    main()