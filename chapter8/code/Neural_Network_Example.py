import matplotlib.pyplot as plot
import numpy as nump
import pandas as pand
#import fix_yahoo_finance as yahf
#import yfinance as yahf
import datetime
import logging
import seaborn as seab

from yahooquery import Ticker



from keras.src.preprocessing.sequence import TimeseriesGenerator
from keras import Sequential,layers


class NeuralNetworkExample:
      
      def __init__(self):
  	      self.name = "NeuralNetworkExample"
      
      def getStockPrices(self,stockSymbol):
    
          stock_data = Ticker(stockSymbol)
          #print("columns ",stock_data.columns)
          
          stock_price_history = stock_data.history(period='max', interval='1d')
          
          return stock_price_history
      
      def create_time_series(self,stock_data, length_val):
          stock_close = stock_data['close']
          stock_dividends = stock_data['dividends']
          stock_time_series_gen = TimeseriesGenerator(stock_close, stock_close,
                              length=length_val,
                              batch_size=len(stock_close))
          stock_global_index = length_val
          index, time = stock_time_series_gen[0]
          stock_has_dividends = nump.zeros(len(index))
          for stock_b_row in range(len(time)):
              #assert(abs(t[b_row] - close[global_index]) <= 0.001)
              stock_has_dividends[stock_b_row] = stock_dividends[stock_global_index] > 0            
              stock_global_index += 1
          return nump.concatenate((index, nump.transpose([stock_has_dividends])),
                           axis=1), time

      def get_neural_network_model(self,layers_n):
          stock_model = Sequential()
          stock_model.add(layers.Dense(64, activation='relu', input_shape=(layers_n+1,)))
          stock_model.add(layers.Dense(64, activation='relu'))
          stock_model.add(layers.Dense(1))
          return stock_model


      def get_model_stock_inputs(self,data, start, end, epochs):
          stock_models = {}
          for inputs in range(start, end+1):
              print('Using {} inputs'.format(inputs))
              model_inputs, targets = self.create_time_series(data, inputs)
        
              train_inputs = model_inputs[:-1000]
              val_inputs = model_inputs[-1000:]
              train_targets = targets[:-1000]
              val_targets = targets[-1000:]
        
              model = self.get_neural_network_model(inputs)
              print('Training')
              model.compile(optimizer='adam', loss='mse') 
              h = model.fit(train_inputs, train_targets,
                      epochs=epochs,
                      batch_size=32,
                      validation_data=(val_inputs, val_targets))
              model_info = {'model': model, 'history': h.history}
              stock_models[inputs] = model_info
          return stock_models

      def plot_stock_model_function(self,stock_model_stats):
          stock_val_loss = []
          stock_indices = []
          for stock_k, stock_v in stock_model_stats.items():
              stock_indices.append(stock_k)
              stock_val_loss.append(stock_v['val_loss'])
          plot.plot(stock_indices, stock_val_loss)
          plot.show()

		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = NeuralNetworkExample()
    


    stock_symbol = "MSFT"


    #period=1
    stock_data = priceIndicator.getStockPrices(stock_symbol)
    print("stock data",stock_data)

    stock_inputs, stock_targets = priceIndicator.create_time_series(stock_data, 4)

    print(stock_inputs[3818])

    stock_h_min = stock_data.min()
    stock_normalized_h = (stock_data - stock_h_min) / (stock_data.max() - stock_h_min)


    stock_inputs, stock_targets = priceIndicator.create_time_series(stock_normalized_h, 4)

    print(stock_inputs[3818])

    stock_train_inputs = stock_inputs[:-1000]
    stock_val_inputs = stock_inputs[-1000:]
    stock_train_targets = stock_targets[:-1000]
    stock_val_targets = stock_targets[-1000:]

    stock_trained_models = priceIndicator.get_model_stock_inputs(stock_normalized_h, 2, 10, 20)


    stock_model_stats = {}
    for stock_k, stock_v in stock_trained_models.items():
        stock_train_history = stock_v['history']
        stock_loss = stock_train_history['loss'][-1]
        stock_val_loss = stock_train_history['val_loss'][-1]
        stock_model_stats[stock_k] = {'inputs': stock_k, 'loss': stock_loss, 'val_loss': stock_val_loss}
	

    priceIndicator.plot_stock_model_function(stock_model_stats)

    stock_close_min = stock_data['close'].min()
    stock_close_max = stock_data['close'].max()
    for stock_k in stock_model_stats:
        stock_e = ((stock_close_max - stock_close_min) * stock_model_stats[stock_k]['val_loss'] + stock_close_min)
        print(stock_k, stock_e)
    


if __name__ == '__main__':
    main()