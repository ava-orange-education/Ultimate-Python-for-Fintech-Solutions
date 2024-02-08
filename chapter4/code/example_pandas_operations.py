import pandas as pand


def create_series():
    stock_prices = pand.Series([23,24,25,26,24])
    days = ["1","2","3","4","5"]
  
    stock_prices.index = days
  
    print(stock_prices)
  
    return stock_prices

def calculate_stats(stock_prices):
    
    mean = stock_prices.mean(skipna = True)
    
    print(" Mean of the stock prices is :", mean)
    
    
    result = stock_prices.apply(lambda x : True if x > mean else False)
    
    print(" applying the condition for time series greater than mean :", result)

if __name__ == "__main__":
   
   stock_prices = create_series()
   
   calculate_stats(stock_prices)