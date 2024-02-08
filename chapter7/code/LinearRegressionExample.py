import matplotlib.pyplot as plot
import numpy as nump
import pandas as pand
import fix_yahoo_finance as yahf
import datetime
import logging



from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split

class LinearRegressionExample:
      
      def __init__(self):
  	      self.name = "LinearRegressionExample"
      
      def getStockPrices(self,stockSymbol,range_start,range_end):
    
          print("getting stock price")
          stock_data=yahf.download(stockSymbol,start=range_start,end=range_end)
      
      
          print("data",stock_data.columns)
      
          return stock_data
      def analyzeRegressionModel(self,stock_train_x,stock_train_y,stock_test_x,stock_test_y):
           regression = LinearRegression()
           regression.fit(stock_train_x, stock_train_y)
           print("regression equation coefficient",regression.coef_)
           print("regression equation intercept",regression.intercept_)

           regression_confidence = regression.score(stock_test_x, stock_test_y)
           print("linear regression model confidence: ", regression_confidence)

           predicted=regression.predict(stock_test_x)
           print(stock_test_x.head())

           stock_data=pand.DataFrame({'Actual_Price':stock_test_y, 'Predicted_Price':predicted})
           print("Stock Actual Price vs Stock Predicted Price", stock_data.head(10))

           print('Stock Mean Absolute Error (MAE):', metrics.mean_absolute_error(stock_test_y, predicted))
           print('Stock Mean Squared Error (MSE) :', metrics.mean_squared_error(stock_test_y, predicted))
           print('Stock Root Mean Squared Error (RMSE):', nump.sqrt(metrics.mean_squared_error(stock_test_y, predicted)))

           stock_x2 =  stock_data.Actual_Price.mean()
           stock_y2 =  stock_data.Predicted_Price.mean()
           Accuracy1 = (stock_x2/stock_y2)*100
           print("The accuracy of the linear regression model is " , Accuracy1)

           plot.scatter( stock_data.Actual_Price,  stock_data.Predicted_Price,  color='Darkblue')
           plot.xlabel("Actual Price")
           plot.ylabel("Predicted Price")
           plot.show()

      def plotStockPrices(self,stock_x, stock_y, title="", xlabel='Date', ylabel='Value', dpi=100):
          plot.figure(figsize=(16,5), dpi=dpi)
          plot.plot(stock_x, stock_y, color='tab:red')
          plot.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
          plot.show()
      
		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = LinearRegressionExample()
    
    start_date = datetime.date(2021,6,21)
    end_date = datetime.date(2022,12,25)
    stock_symbol = "MSFT"
    period=1
    training_data = priceIndicator.getStockPrices(stock_symbol,start_date,end_date)
    print("stock data",training_data)

    stock_x = training_data.index
    stock_y = training_data['Close']
    #df.columns ['date','open','high','low','close','vol','divs','split']


    title = (stock_symbol,"History stock performance till date")
    priceIndicator.plotStockPrices( stock_x , stock_y , title=title,xlabel='Date', ylabel='Value',dpi=100)

    training_data.reset_index(inplace=True)

    stock_x = training_data[['Open', 'High','Low', 'Volume']]
    stock_y = training_data['Close']

    stock_train_x, stock_test_x, stock_train_y, stock_test_y = train_test_split(stock_x,stock_y, test_size=0.15 , shuffle=False,random_state = 0)


    priceIndicator.analyzeRegressionModel(stock_train_x,stock_train_y,stock_test_x,stock_test_y)


    


if __name__ == '__main__':
    main()