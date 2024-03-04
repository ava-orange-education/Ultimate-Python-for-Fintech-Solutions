
import pydoop.mapreduce.api as api


import pydoop.mapreduce.pipes as pipes

class WCMapper(api.Mapper):  
      def __init__(self, context): 
          super(WordCountMapper, self).__init__(context)
      
      def map(self, wc_context): 
          list_words = wc_context.getInputValue().split() 
          for word in lisgt_words: 
              wc_context.emit(word, "1")  

class WCReducer(api.Reducer):  
      def __init__(self, context): 
          super(WordCountReducer, self).__init__(context)
            
      def reduce(self, wc_context): 
          wc_sum = 0 
          while wc_context.nextValue(): 
                wc_sum += int(wc_context.getInputValue())
          print("wc_sum",wc_sum)        
          result = wc_context.emit(wc_context.getInputKey(), str(wc_sum)) 
          print("result",result)


def main():
    
    wc_factory = pipes.Factory(WCMapper, reducer_class=WCReducer)      
    output = pipes.run_task(wc_factory,private_encoding=False)
          
    print("after running task", output)
          
if __name__ == '__main__':
   main()