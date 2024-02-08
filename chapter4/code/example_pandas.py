
import pandas as pand


def read_data(path):
  
  dataf = pand.read_csv(path)
  print(dataf)
  print(pand.options.display.max_rows) 
  print(dataf.loc[0])
  print(dataf.loc[[0, 1]])
  
  new_data = {
    "steps": [420, 380, 390],
    "time": [50, 40, 45]
  }
  
  dataf = pand.DataFrame(new_data, index = ["first", "second", "third"])
  
  print(dataf) 
  print(dataf.loc["second"])
  

if __name__ == "__main__":

   path = "data.csv"

   read_data(path)