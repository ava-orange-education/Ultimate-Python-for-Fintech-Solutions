import numpy as np 
import pandas as pd

import matplotlib.pyplot as plot
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as plotgo

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as tornn

import datetime


import math, time
from sklearn.metrics import mean_squared_error

from pytorch_gru import PyTorchGRU
import yfinance as yfin

import pandas_datareader as data_reader

import os
import logging


class PyTorchAssetPricePredictor(tornn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PyTorchAssetPricePredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = tornn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = tornn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
        
def plot_results(result):
    
   fig = plotgo.Figure()
   fig.add_trace(plotgo.Scatter(plotgo.Scatter(x=result.index, y=result[0],
                       mode='lines',
                       name='Trained Data prediction')))
   fig.add_trace(plotgo.Scatter(x=result.index, y=result[1],
                       mode='lines',
                       name='Testing Data prediction'))
   fig.add_trace(plotgo.Scatter(plotgo.Scatter(x=result.index, y=result[2],
                       mode='lines',
                       name='Actual Asset Value')))
   fig.update_layout(
       xaxis=dict(
           showline=True,
           showgrid=True,
           showticklabels=False,
           linecolor='white',
           linewidth=2
       ),
       yaxis=dict(
           title_text='Close (USD)',
           titlefont=dict(
               family='Rockwell',
               size=12,
               color='white',
           ),
           showline=True,
           showgrid=True,
           showticklabels=True,
           linecolor='white',
           linewidth=2,
           ticks='outside',
           tickfont=dict(
               family='Rockwell',
               size=12,
               color='white',
           ),
       ),
       showlegend=True,
       template = 'plotly_dark'

   )



   annotations = []
   annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                                 xanchor='left', yanchor='bottom',
                                 text='AssetPrice Results',
                                 font=dict(family='Rockwell',
                                           size=26,
                                           color='white'),
                                 showarrow=False))
   fig.update_layout(annotations=annotations)

   fig.show()



def split_data(stock, lookback):
    data_raw = stock.to_numpy() 
    data = []


    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])

    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);

    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]

    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]

    return [x_train, y_train, x_test, y_test]
    
def get_data(stock_symbol,start_date):
    yfin.pdr_override()
    
    dataset = data_reader.data.get_data_yahoo(stock_symbol,start=start_date)

    #close = dataset['Close']

    return dataset
    
def main():
    logging.basicConfig(
        format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("ai_pytorch").setLevel(logging.INFO)
    lookback = 20 
    
    start_date = datetime.date(2021,6,21)
    end_date = datetime.date(2023,8,9)
    stock_symbol = "MSFT"
    
    data = get_data(stock_symbol,start_date)
    
    print("data",data)
    
    price = data[['Close']]
    print("Close Price :",price.info())



    scaler = MinMaxScaler(feature_range=(-1, 1))
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
    
    x_train, y_train, x_test, y_test = split_data(price, lookback)
    print('x_train.shape = ',x_train.shape)
    print('y_train.shape = ',y_train.shape)
    print('x_test.shape = ',x_test.shape)
    print('y_test.shape = ',y_test.shape)


    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
    y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)



    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    num_epochs = 100
    
    model = PyTorchAssetPricePredictor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []

    for t in range(num_epochs):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train_lstm)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
    training_time = time.time()-start_time
    print("Training time: {}".format(training_time))


    predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
    original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))
    

    y_test_pred = model(x_test)

    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train_lstm.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test_lstm.detach().numpy())

    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    print('Trained Score is: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    print('Testing Score is: %.2f RMSE' % (testScore))
    lstm.append(trainScore)
    lstm.append(testScore)
    lstm.append(training_time)


    trainPredictPlot = np.empty_like(price)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred

    testPredictPlot = np.empty_like(price)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(y_train_pred)+lookback-1:len(price)-1, :] = y_test_pred

    original = scaler.inverse_transform(price['Close'].values.reshape(-1,1))

    predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
    predictions = np.append(predictions, original, axis=1)
    result = pd.DataFrame(predictions)
    

        

    model = PyTorchGRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)


    hist = np.zeros(num_epochs)
    start_time = time.time()
    gru = []

    for t in range(num_epochs):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train_gru)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    training_time = time.time()-start_time    
    print("Training time is: {}".format(training_time))


    predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
    original = pd.DataFrame(scaler.inverse_transform(y_train_gru.detach().numpy()))
    
    y_test_pred = model(x_test)

    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train_gru.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test_gru.detach().numpy())

    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    print('Trained Score is: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    print('Testing Score is: %.2f RMSE' % (testScore))
    gru.append(trainScore)
    gru.append(testScore)
    gru.append(training_time)
    
    trainPredictPlot = np.empty_like(price)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred


    testPredictPlot = np.empty_like(price)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(y_train_pred)+lookback-1:len(price)-1, :] = y_test_pred

    original = scaler.inverse_transform(price['Close'].values.reshape(-1,1))

    predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
    predictions = np.append(predictions, original, axis=1)
    result = pd.DataFrame(predictions)
    
    
    plot_results(result)
    
    

if __name__ == '__main__':
    main()

