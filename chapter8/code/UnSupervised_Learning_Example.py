import matplotlib.pyplot as plot
import numpy as nump
import pandas as pand
#import fix_yahoo_finance as yahf
#import yfinance as yahf
import datetime
import logging
import seaborn as seab

from yahooquery import Ticker

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans,DBSCAN

#from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class UnSupervisedLearningExample:
      
      def __init__(self):
  	      self.name = "UnSupervisedLearningExample"
      
      def getStockPrices(self,stockSymbols):
    
          
          dtf = pand.DataFrame(columns=['ROE(%)','Beta'])
          stock_symbols_remove = []
          for stockSymbol in stockSymbols:
              print("getting stock price",stockSymbol)
              #stock_data = yahf.Ticker(stockSymbol)
              stock_data = Ticker(stockSymbol)
              print("stock_data",stock_data)
              

              try:
                 #print("getting stock_ROE")
                 #print("net income",stock_data.income_statement()['NetIncome'])
                 #print("equity",stock_data.balance_sheet()['StockholdersEquity'])
                 stock_ROE = stock_data.all_financial_data()['NetIncome']/stock_data.all_financial_data()['StockholdersEquity']*100
                 #print(stockSymbol," :stock ROE",stock_ROE)
                 stock_Mean_ROE = pand.Series(stock_ROE.mean())
                 #print("stock mean:",stock_Mean_ROE)
                 stock_Beta = pand.Series(stock_data.key_stats[stockSymbol]['beta'])

                 #print("stock beta: ",stock_Beta)

                 stock_values_to_add = {'ROE(%)': stock_Mean_ROE.values[0].round(2), 'Beta': stock_Beta.values[0].round(2)}
                 stock_row_to_add = pand.Series(stock_values_to_add,name=stockSymbol)
                 #print("stock data added",stock_row_to_add)
                 #dtf = pand.concat([dtf,stock_row_to_add],ignore_index=F) 
                 dtf.loc[len(dtf)]  = stock_row_to_add

              except Exception as exception:
                 
                     print("exception :",exception)
                 
                     stock_symbols_remove.append(stockSymbol)
          dtf["stock"] = stockSymbols
          print("stock symbols removed",stock_symbols_remove)
          dtf =  dtf.set_index("stock")
          print("data",dtf)
      
          return dtf
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

      def plotKMeans(self,stock_data):
		  
          
          plot.figure(figsize=(12, 8))

		  
          seab.set_style('whitegrid')

          #dbscan = DBSCAN()
          #clusters = dbscan.fit(stock_data.to_numpy())
		  
          seax = seab.scatterplot(y="ROE(%)", x="Beta", edgecolor='face', hue="cluster",data=stock_data, palette = 'bright',s=60)

           #seab.scatterplot(data=stock_data,)
		  
          plot.xlabel('Beta', size=17)
		  
          plot.ylabel('ROE(%)', size=17)
		  
          plot.setp(seax.get_legend().get_texts(), fontsize='17')
		  
          plot.setp(seax.get_legend().get_title(), fontsize='17')
		  
          plot.title('CLUSTERS from k-means algorithm with k = 4',fontsize='x-large')

         
          

          for i in range(0,stock_data.shape[0]):
		      
              plot.text(stock_data["Beta"][i]+0.07, stock_data['ROE(%)'][i]+0.01, stock_data.index[i],horizontalalignment='right',verticalalignment='bottom', size='small', color='black', weight='semibold')

          plot.show()
          
          '''		  
          stock_inertia = []
		  
          stock_k_range = range(1,10)
		  
          for k in stock_k_range:
		  
              model = KMeans(n_clusters=k)
		  
              model.fit(stock_data)
		  
              stock_inertia.append(model.inertia_)


		  
          plot.figure(figsize=(15,5))
		  
          plot.xlabel('k value',fontsize='x-large')
		  
          plot.ylabel('Model inertia',fontsize='x-large')
		  
          plot.plot(stock_k_range,stock_inertia,color='r')
		  
          plot.show()
          '''

      
		  
def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = UnSupervisedLearningExample()
    
    #start_date = datetime.date(2021,6,21)
    #end_date = datetime.date(2022,12,25)

    #stock_symbols = ["ADBE","AEP","CSCO","EXC","INTC","LNT","STLD","TMUS","XEL"]

    stock_symbols = ["MSFT","AAPL","IBM","GE","FORD","DELL"]
    #stock_symbols.append("MSFT")
    #stock_symbols.append("AAPL")
    #stock_symbols.append("IBM")
    #stock_symbols.append("GE")

    period=1
    stock_data = priceIndicator.getStockPrices(stock_symbols)
    print("stock data",stock_data)

    data_frame_stack = stock_data.copy()

	
    
    scaler = StandardScaler()
	
    stock_data_values = scaler.fit_transform(data_frame_stack.values)

	
    print("scaled data values",stock_data_values)
		
	
    kmeans_stock_model = KMeans(n_clusters=2).fit(stock_data_values)

    stock_clusters = kmeans_stock_model.labels_

    stock_data['cluster']=stock_clusters

    print("data - cluster",stock_data)
		
    priceIndicator.plotKMeans(stock_data)


    


if __name__ == '__main__':
    main()