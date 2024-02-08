import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import fix_yahoo_finance as yahf
import datetime
import logging


class RSIStockPrice:
  
      def __init__(self):
  	      self.name = "RSIStockPrice"
  	      self.data = None
  	      self.dataset_index = 0
          
      def getStockPositions(self,data,num,window1,window2):

          data['rsi']=0.0
          data['rsi'][num:]=self.relativeStrengthIndex(data['Close'],num)

          data['positions']=np.select([data['rsi']<window1,data['rsi']>window2], \
                                    [1,-1],default=0)
          data['signals']=data['positions'].diff()

          return data[num:]
        
      def relativeStrengthIndex(self,data,num):

          deltaDiff=data.diff().dropna()

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
        
      def createPlot(self,data,stock_symbol):

          fig=plot.figure(figsize=(10,10))
          axis=fig.add_subplot(211)

          data['Close'].plot(label=stock_symbol)
          axis.plot(data.loc[data['signals']==1].index,
                  data['Close'][data['signals']==1],
                  label='LONG',lw=0,marker='^',c='g')
          axis.plot(data.loc[data['signals']==-1].index,
                  data['Close'][data['signals']==-1],
                  label='SHORT',lw=0,marker='v',c='r')


          plot.legend(loc='best')
          plot.grid(True)
          plot.title('Positions')
          plot.xlabel('Date')
          plot.ylabel('price')

          plot.show()

          baxis=plot.figure(figsize=(10,10)).add_subplot(212,sharex=axis)
          data['rsi'].plot(label='relative strength index',c='#522e75')
          baxis.fill_between(data.index,30,70,alpha=0.5,color='#f22f08')

          baxis.text(data.index[-45],75,'overbought',color='#594346',size=12.5)
          baxis.text(data.index[-45],25,'oversold',color='#594346',size=12.5)

          plot.xlabel('Date')
          plot.ylabel('value')
          plot.title('RSI')
          plot.legend(loc='best')
          plot.grid(True)
          plot.show()

      def getStockPrices(self,stockSymbol,range_start,range_end):
    
          print("getting stock price")
          data=yahf.download(stockSymbol,start=range_start,end=range_end)
      
      
          print("data",data)
      
          return data



def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
    logging.getLogger("finance").setLevel(logging.INFO)
    priceIndicator = RSIStockPrice()
    
    start_date = datetime.date(2021,6,21)
    end_date = datetime.date(2023,8,9)
    stock_symbol = "MSFT"
    period=1
    data = priceIndicator.getStockPrices(stock_symbol,start_date,end_date)
    print("stock data",data)
    num = 14
    window1 = 30
    window2 = 70
    positions = priceIndicator.getStockPositions(data,num,window1,window2)
    priceIndicator.createPlot(positions,stock_symbol)
    
    

if __name__ == '__main__':
    main()