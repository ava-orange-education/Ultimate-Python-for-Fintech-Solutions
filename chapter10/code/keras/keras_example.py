

import numpy as np
import time as tm
import datetime as dt



from yahoo_fin import stock_info as yahoof
from sklearn.preprocessing import MinMaxScaler
from collections import deque

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

import matplotlib.pyplot as plot


import logging



class KerasAssetPricePredictor:
    
    
    def __init__(self):
        self.name = "Keras"
        self.data = None
        self.dataset_index = 0

    def createPlot(self,data_frame,stockSymbol):
        plot.style.use(style='ggplot')
        plot.figure(figsize=(16,10))
        plot.plot(data_frame['close'][-200:])
        plot.xlabel('days')
        plot.ylabel('price')
        plot.legend([f'Actual price for {stockSymbol}'])
        plot.show()



    def generateData(self,num_days,data_frame,num_steps):
      dataf = data_frame.copy()
      dataf['future'] = dataf['scaled_close'].shift(-num_days)
      last_sequence = np.array(dataf[['scaled_close']].tail(num_days))
      dataf.dropna(inplace=True)
      sequence_data = []
      sequences = deque(maxlen=num_steps)

      for entry, target in zip(dataf[['scaled_close'] + ['date']].values, dataf['future'].values):
          sequences.append(entry)
          if len(sequences) == num_steps:
              sequence_data.append([np.array(sequences), target])

      last_sequence = list([s[:len(['scaled_close'])] for s in sequences]) + list(last_sequence)
      last_sequence = np.array(last_sequence).astype(np.float32)


      X, Y = [], []
      for seq, target in sequence_data:
          X.append(seq)
          Y.append(target)

      X = np.array(X)
      Y = np.array(Y)

      return dataf, last_sequence, X, Y


    def trainPredictorModel(self,x_train, y_train,num_steps):
      model = Sequential()
      model.add(LSTM(60, return_sequences=True, input_shape=(num_steps, len(['scaled_close']))))
      model.add(Dropout(0.3))
      model.add(LSTM(120, return_sequences=False))
      model.add(Dropout(0.3))
      model.add(Dense(20))
      model.add(Dense(1))

      batch_size = 8
      num_epochs = 80

      model.compile(loss='mean_squared_error', optimizer='adam')

      model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=num_epochs,
                verbose=1)

      model.summary()

      return model

    def plotPredictions(self,data,search_steps,num_days,num_steps,stockSymbol,scaler):
        predictions = []

        for step in search_steps:
          df, last_sequence, x_train, y_train = self.generateData(num_days,data,num_steps)
          x_train = x_train[:, :, :len(['scaled_close'])].astype(np.float32)

          model = self.trainPredictorModel(x_train, y_train,num_steps)

          last_sequence = last_sequence[-num_steps:]
          last_sequence = np.expand_dims(last_sequence, axis=0)
          prediction = model.predict(last_sequence)
          predicted_price = scaler.inverse_transform(prediction)[0][0]

          predictions.append(round(float(predicted_price), 2))

        if bool(predictions) == True and len(predictions) > 0:
          predictions_list = [str(d)+'$' for d in predictions]
          predictions_str = ', '.join(predictions_list)
          message = f'{stockSymbol} prediction for upcoming 3 days ({predictions_str})'

          print(message)

        copy_df = data.copy()
        y_predicted = model.predict(x_train)
        y_predicted_transformed = np.squeeze(scaler.inverse_transform(y_predicted))
        first_seq = scaler.inverse_transform(np.expand_dims(y_train[:6], axis=1))
        last_seq = scaler.inverse_transform(np.expand_dims(y_train[-3:], axis=1))
        y_predicted_transformed = np.append(first_seq, y_predicted_transformed)
        y_predicted_transformed = np.append(y_predicted_transformed, last_seq)
        copy_df[f'predicted_close'] = y_predicted_transformed


        date_now = dt.date.today()
        date_tomorrow = dt.date.today() + dt.timedelta(days=1)
        date_after_tomorrow = dt.date.today() + dt.timedelta(days=2)

        copy_df.loc[date_now] = [predictions[0], f'{date_now}', 0, 0]
        copy_df.loc[date_tomorrow] = [predictions[1], f'{date_tomorrow}', 0, 0]
        copy_df.loc[date_after_tomorrow] = [predictions[2], f'{date_after_tomorrow}', 0, 0]

        plot.style.use(style='ggplot')
        plot.figure(figsize=(16,10))
        plot.plot(copy_df['close'][-150:].head(147))
        plot.plot(copy_df['predicted_close'][-150:].head(147), linewidth=1, linestyle='dashed')
        plot.plot(copy_df['close'][-150:].tail(4))
        plot.xlabel('days')
        plot.ylabel('price')
        plot.legend([f'Actual Asset price - {stockSymbol}',
                    f'Predicted Asset price - {stockSymbol}',
                    f'Predicted Asset price after 3 days'])
        plot.show()

def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    pricePredictor = KerasAssetPricePredictor()
    
    num_steps = 7

    search_steps = [1, 2, 3]

    stock_symbol = 'MSFT'
    
    date_now = tm.strftime('%Y-%m-%d')
    date_3_years_back = (dt.date.today() - dt.timedelta(days=1104)).strftime('%Y-%m-%d')
    
    data_frame = yahoof.get_data(
        stock_symbol,
        start_date=date_3_years_back,
        end_date=date_now,
        interval='1d')


    data_frame = data_frame.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
    data_frame['date'] = data_frame.index
    
    pricePredictor.createPlot(data_frame,stock_symbol)
    
    scaler = MinMaxScaler()
    data_frame['scaled_close'] = scaler.fit_transform(np.expand_dims(data_frame['close'].values, axis=1))
    
    num_days = 3
    
    pricePredictor.plotPredictions(data_frame,search_steps,num_days,num_steps,stock_symbol,scaler)
    
    

if __name__ == '__main__':
    main()