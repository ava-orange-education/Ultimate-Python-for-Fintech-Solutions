import math
import random
import numpy as nump
#import pandas as pand
import tensorflow as tensorf
import matplotlib.pyplot as plot
import pandas_datareader as data_reader
import logging
import datetime
import yfinance as yfin

from tqdm import tqdm_notebook, tqdm
from collections import deque



class TensorFlowAssetPricePredictor():

    def __init__(self, state_size, action_space=3, model_name="TensorFlow"):
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.inventory = []
        self.model_name = model_name
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995
        self.model = self.create_model()

    def create_model(self):
        model = tensorf.keras.models.Sequential()
        model.add(tensorf.keras.layers.Dense(
            units=32, activation='relu', input_dim=self.state_size))
        model.add(tensorf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tensorf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tensorf.keras.layers.Dense(
            units=self.action_space, activation='linear'))
        model.compile(
            loss='mse', optimizer=tensorf.keras.optimizers.Adam(lr=0.001))

        return model

    def transact(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)

        actions = self.model.predict(state)
        
        return nump.argmax(actions[0])

    def batch_train(self, batch_size):

        batch = []
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])

        for state, action, reward, next_state, done in batch:
            reward = reward
            if not done:
                reward = reward + self.gamma * \
                    nump.amax(self.model.predict(next_state)[0])

            target = self.model.predict(state)
            target[0][action] = reward

            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

    def get_sigmoid(self,y):
        return 1/(1 + math.exp(-y))

    def stock_price_format(n):
        if n < 0:
            return "- # {0:2f}".format(abs(n))
        else:
            return "$ {0:2f}".format(abs(n))

    def get_data(self,stock_symbol,start_date):

        #dataset = data_reader.DataReader(stock_symbol, data_source="yahoo")
        
        yfin.pdr_override()
        
        dataset = data_reader.data.get_data_yahoo(stock_symbol,start=start_date)

        #start_date = str(dataset.index[0]).split()[0]
        #end_date = str(dataset.index[1]).split()[0]

        close = dataset['Close']

        return close

    def create_state(self,data, timestep, window_size):

        starting_id = timestep - window_size + 1
        
        #print("starting_id",starting_id)

        if starting_id >= 0:
            windowed_data = data[starting_id:timestep+1]
        else:
            windowed_data = - starting_id * [data[0]] + list(data[0:timestep+1])
            #windowed_data = list(data[0:timestep+1])
            
        print("windowed data",windowed_data)

        state = []
        for i in range(window_size - 1):
            #print("i=",i)
            state.append(self.get_sigmoid(windowed_data[i+1] - windowed_data[i]))

        return nump.array([state])
        
    def format_asset_price(self,n):
      if n < 0:
        return "- $ {0:2f}".format(abs(n))
      else:
        return "$ {0:2f}".format(abs(n))


def main():
    logging.basicConfig(
        format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("ai_tensorflow").setLevel(logging.INFO)

    window_size = 10
    epochs = 1000
    
    start_date = datetime.date(2021,6,21)
    end_date = datetime.date(2023,8,9)
    

    assetPricePredictor = TensorFlowAssetPricePredictor(window_size)
    
    #print("model summary",assetPricePredictor.model)
    print("model summary",assetPricePredictor.model.summary(expand_nested=True))

    data = assetPricePredictor.get_data("MSFT",start_date)
    batch_size = 32
    data_samples = len(data) - 1

    print("length of data",data)

    for epoch in range(1, epochs + 1):

        print("Current epoch: {}/{}".format(epoch, epochs))

        state = assetPricePredictor.create_state(data, 0, window_size + 1)

        total_profit = 0
        assetPricePredictor.inventory = []

        for t in tqdm(range(data_samples)):

            action = assetPricePredictor.transact(state)

            next_state = assetPricePredictor.create_state(data, t+1, window_size + 1)
            reward = 0

            if action == 1: 
                assetPricePredictor.inventory.append(data[t])
                print("Asset is bought: ", assetPricePredictor.format_asset_price(data[t]))

            elif action == 2 and len(assetPricePredictor.inventory) > 0:
                buy_price = assetPricePredictor.inventory.pop(0)

                reward = max(data[t] - buy_price, 0)
                total_profit += data[t] - buy_price
                print("Asset is sold: ", assetPricePredictor.format_asset_price(
                    data[t]), " Profit from this transaction: " + assetPricePredictor.format_asset_price(data[t] - buy_price))

            if t == data_samples - 1:
                done = True
            else:
                done = False

            assetPricePredictor.memory.append(
                (state, action, reward, next_state, done))

            state = next_state

            if done:
                print("Total profit from this transaction is {}".format(
                    total_profit))

            if len(assetPricePredictor.memory) > batch_size:
                assetPricePredictor.batch_train(batch_size)

        if epoch % 10 == 0:
            assetPricePredictor.model.save(
                "assetPricePredictor_{}.h5".format(epoch))


if __name__ == '__main__':
    main()
