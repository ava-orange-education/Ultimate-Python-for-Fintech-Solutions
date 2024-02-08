import pyfolio as pyfolio
import warnings

warnings.filterwarnings('ignore')

def create_pyfolio_returns(stock_returns,start_date):
    pyfolio.create_returns_tear_sheet(stock_returns, live_start_date=start_date)



if __name__ == "__main__":
   stock_returns = pyfolio.utils.get_symbol_rets('FB')
   
   
   start_date='2015-12-1'
   
   
   create_pyfolio_returns(stock_returns,live_start_date=start_date)