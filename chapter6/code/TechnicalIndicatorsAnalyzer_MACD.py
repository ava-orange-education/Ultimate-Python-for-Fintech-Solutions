import matplotlib.pyplot as plot
import numpy as np
import pandas as pand
import fix_yahoo_finance as yahf
import datetime
import logging
from ta.utils import dropna
from ta.volatility import BollingerBands

class TechnicalIndicatorsAnalyzer:
  
      def __init__(self):
  	      self.name = "TechnicalIndicatorsAnalyzer"
  	      self.data = None
  	      self.dataset_index = 0
          
          
      def getRSIStockPositions(self,stock_data,num,window1,window2):

          stock_data['rsi']=0.0
          stock_data['rsi'][num:]=self.relativeStrengthIndex(stock_data['Close'],num)

          stock_data['positions']=np.select([stock_data['rsi']<window1,stock_data['rsi']>window2], \
                                    [1,-1],default=0)
          stock_data['signals']=stock_data['positions'].diff()

          return stock_data[num:]
        
      def relativeStrengthIndex(self,stock_data,num):

          deltaDiff=stock_data.diff().dropna()

          upward=np.where(deltaDiff>0,deltaDiff,0)
          downward=np.where(deltaDiff<0,-deltaDiff,0)

          up_average = self.smoothedMovingAverage(upward,num)
          down_average = self.smoothedMovingAverage(downward,num)
          relative_strength=np.divide(up_average,down_average)

          output=100-100/(1+relative_strength)

          return output[num-1:]

        
        
      def smoothedMovingAverage(self,series,num):

          output=[series[0]]

          for i in range(1,len(series)):
              temp=output[-1]*(num-1)+series[i]
              output.append(temp/num)
    
          return output
          
      def getSimpleMovingAverage(self,stock_data, ndays): 
              SMA = pand.Series(stock_stock_data['Close'].rolling(ndays).mean(), name = 'SMA') 
              stock_data = stock_stock_data.join(SMA) 
              return stock_data

      def getExponentiallyWeightedMovingAverage(self,stock_data, ndays): 
              EMA = pand.Series(stock_stock_data['Close'].ewm(span = ndays, min_periods = ndays - 1).mean(), 
                           name = 'EWMA_' + str(ndays)) 
              stock_data = stock_stock_data.join(EMA) 
              return stock_data
              
      def plotSMA_EWMA(self,stock_data,num_days,ew_num):
          
          stock_SMA = self.getSimpleMovingAverage(stock_data,num_days)
          stock_SMA = stock_SMA.dropna()
          stock_SMA = stock_SMA['SMA']


          stock_EWMA = self.getExponentiallyWeightedMovingAverage(stock_data,ew_num)
          stock_EWMA = stock_EWMA.dropna()
          stock_EWMA = stock_EWMA['EWMA_200']
          

          plot.figure(figsize=(10,7))


          plot.title('Moving Average')
          plot.xlabel('Date')
          plot.ylabel('Price')


          plot.plot(stock_stock_data['Close'],lw=1, label='Close Price')
          plot.plot(stock_SMA,'g',lw=1, label='50-day SMA')
          plot.plot(stock_EWMA,'r', lw=1, label='200-day EMA')


          plot.legend()

          plot.show()
        
      def createPlot(self,data,stock_symbol):

          fig=plot.figure(figsize=(10,10))
          axis=fig.add_subplot(211)

          stock_data['Close'].plot(label=stock_symbol)
          axis.plot(stock_data.loc[stock_data['signals']==1].index,
                  stock_data['Close'][stock_data['signals']==1],
                  label='LONG',lw=0,marker='^',c='g')
          axis.plot(stock_data.loc[stock_data['signals']==-1].index,
                  stock_data['Close'][stock_data['signals']==-1],
                  label='SHORT',lw=0,marker='v',c='r')


          plot.legend(loc='best')
          plot.grid(True)
          plot.title('Positions')
          plot.xlabel('Date')
          plot.ylabel('price')

          plot.show()

          baxis=plot.figure(figsize=(10,10)).add_subplot(212,sharex=axis)
          stock_data['rsi'].plot(label='relative strength index',c='#522e75')
          baxis.fill_between(stock_data.index,30,70,alpha=0.5,color='#f22f08')

          baxis.text(stock_data.index[-45],75,'overbought',color='#594346',size=12.5)
          baxis.text(stock_data.index[-45],25,'oversold',color='#594346',size=12.5)

          plot.xlabel('Date')
          plot.ylabel('value')
          plot.title('RSI')
          plot.legend(loc='best')
          plot.grid(True)
          plot.show()

      def getStockPrices(self,stockSymbol,range_start,range_end):
    
          print("getting stock price")
          stock_data=yahf.download(stockSymbol,start=range_start,end=range_end)
      
      
          print("data",stock_data.columns)
      
          return stock_data
          
      def createRSIPlot(self,stock_data,stock_symbol):

          fig=plot.figure(figsize=(10,10))
          axis=fig.add_subplot(211)

          stock_data['Close'].plot(label=stock_symbol)
          axis.plot(stock_data.loc[stock_data['signals']==1].index,
                  stock_data['Close'][stock_data['signals']==1],
                  label='LONG',lw=0,marker='^',c='g')
          axis.plot(stock_data.loc[stock_data['signals']==-1].index,
                  stock_data['Close'][stock_data['signals']==-1],
                  label='SHORT',lw=0,marker='v',c='r')


          plot.legend(loc='best')
          plot.grid(True)
          plot.title('Positions')
          plot.xlabel('Date')
          plot.ylabel('price')

          plot.show()

          baxis=plot.figure(figsize=(10,10)).add_subplot(212,sharex=axis)
          stock_data['rsi'].plot(label='relative strength index',c='#522e75')
          baxis.fill_between(stock_data.index,30,70,alpha=0.5,color='#f22f08')

          baxis.text(stock_data.index[-45],75,'overbought',color='#594346',size=12.5)
          baxis.text(stock_data.index[-45],25,'oversold',color='#594346',size=12.5)

          plot.xlabel('Date')
          plot.ylabel('value')
          plot.title('RSI')
          plot.legend(loc='best')
          plot.grid(True)
          plot.show()
		  
      def getMovingAverage(self,prices,window1,window2,period):
	        prices['first']=prices['Close'].rolling(window=window1,min_periods=period,center=False).mean()
	        prices['second']=prices['Close'].rolling(window=window1,min_periods=period,center=False).mean()
  
	        return prices
        
      def getMACDStockPositions(self,data,window1,window2,period):
        
          prices = self.getMovingAverage(data,window1,window2,period)
        
          prices['positions'] = 0
        
          prices['positions'][window1:]=np.where(prices['first'][window1:]>=prices['second'][window1:],1,0)

          prices['prices']=prices['positions'].diff()

          prices['oscillator']=prices['first']-prices['second']

          return prices

      def createMACDPlot(self,prices, stockSymbol):

          fig=plot.figure()
          ax=fig.add_subplot(111)

          prices['Close'].plot(label=stockSymbol)
          ax.plot(prices.loc[prices['prices']==1].index,prices['Close'][prices['prices']==1],label='LONG',lw=0,marker='^',c='g')
          ax.plot(prices.loc[prices['prices']==-1].index,prices['Close'][prices['prices']==-1],label='SHORT',lw=0,marker='v',c='r')

          plot.legend(loc='best')
          plot.grid(True)
          plot.title('Positions')

          plot.show()

          fig=plot.figure()
          cx=fig.add_subplot(211)

          prices['oscillator'].plot(kind='bar',color='r')

          plot.legend(loc='best')
          plot.grid(True)
          plot.xticks([])
          plot.xlabel('')
          plot.title('MACD Oscillator')

          bx=fig.add_subplot(212)

          prices['first'].plot(label='first')
          prices['second'].plot(label='second',linestyle=':')

          plot.legend(loc='best')
          plot.grid(True)
          plot.show()



def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = TechnicalIndicatorsAnalyzer()
    
    start_date = datetime.date(2021,6,21)
    end_date = datetime.date(2023,8,9)
    stock_symbol = "MSFT"
    period=1
    data = priceIndicator.getStockPrices(stock_symbol,start_date,end_date)
    print("stock data",data)
    period = 1
    window1 = 10
    window2 = 18
    
    stock_macd_data = priceIndicator.getMACDStockPositions(data,window1,window2,period)

    
    priceIndicator.createMACDPlot(stock_macd_data,stock_symbol)
    


if __name__ == '__main__':
    main()
