import yfinance as yf
import os
import json
import pandas as pand

import matplotlib.pyplot as plot

import logging
import datetime

from sklearn.metrics import precision_score

import yfinance as yahfin
import pandas_datareader as data_reader

from sklearn.ensemble import RandomForestClassifier
import numpy as nump


class SklearnAssetPricePredictor():
    
      def __init__(self):
          self.name = "Sklearn"
          self.data = None
          self.dataset_index = 0
    
      def get_data(self,stock_symbol,start_date):
          
          ticker = yahfin.Ticker(stock_symbol)
          dataset = ticker.history(period="max")

          return dataset
def backtest(data, model, predictors, start=1000, step=750):
    predictions = []
    for i in range(start, data.shape[0], step):
        print("i ",i)
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        

        model.fit(train[predictors], train["Target"])
        
        preds = model.predict_proba(test[predictors])[:,1]
        preds = pand.Series(preds, index=test.index)
        preds[preds > .6] = 1
        preds[preds<=.6] = 0
        
        combined = pand.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
        
        predictions.append(combined)
        
    print("predictions ",predictions
    )
    
    return pand.concat(predictions)



def main():
    logging.basicConfig(
        format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("ai_sklearn").setLevel(logging.INFO)
    
    assetPricePredictor = SklearnAssetPricePredictor()
    
    start_date = datetime.date(2021,6,21)
    end_date = datetime.date(2023,8,9)
    stock_symbol = "MSFT"
    dataset = assetPricePredictor.get_data(stock_symbol,start_date)
    data = dataset[["Close"]]
    data = data.rename(columns = {'Close':'Actual_Close'})

    #data = dataset
    data["Target"] = dataset.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

    data_prev = dataset.copy()
    data_prev = data_prev.shift(1)

    predictors = ["Close", "Volume", "Open", "High", "Low"]
    data = data.join(data_prev[predictors]).iloc[1:]
    
    print(" data ",data)

    model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)


    train = data.iloc[:-100]
    test = data.iloc[-100:]

    model.fit(train[predictors], train["Target"])

    preds = model.predict(test[predictors])
    preds = pand.Series(preds, index=test.index)
    precision_score(test["Target"], preds)


    combined = pand.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
    combined.plot()
    
    print("predictors ",predictors)
    
    predictions = backtest(data, model, predictors)



    predictions["Predictions"].value_counts()


    precision_score(predictions["Target"], predictions["Predictions"])

    weekly_mean = data.rolling(7).mean()
    quarterly_mean = data.rolling(90).mean()
    annual_mean = data.rolling(365).mean()
    weekly_trend = data.shift(1).rolling(7).mean()["Target"]

    data["weekly_mean"] = weekly_mean["Close"] / data["Close"]
    data["quarterly_mean"] = quarterly_mean["Close"] / data["Close"]
    data["annual_mean"] = annual_mean["Close"] / data["Close"]

    data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
    data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]
    data["weekly_trend"] = weekly_trend

    data["open_close_ratio"] = data["Open"] / data["Close"]
    data["high_close_ratio"] = data["High"] / data["Close"]
    data["low_close_ratio"] = data["Low"] / data["Close"]


    full_predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "open_close_ratio", "high_close_ratio", "low_close_ratio", "weekly_trend"]
    predictions = backtest(data.iloc[365:], model, full_predictors)



    precision_score(predictions["Target"], predictions["Predictions"])

    print("value counts of Predictions ",predictions["Predictions"].value_counts())


    predictions.iloc[-100:].plot()
    
    plot.show()

if __name__ == '__main__':
    main()
