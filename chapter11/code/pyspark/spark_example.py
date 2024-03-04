from pyspark import SparkContext


spark_context = SparkContext("local", "Word Counting Application")
list_words = spark_context.parallelize (
   ["Richard Henderson", 
   "Jack Smith", 
   "Harry Smith", 
   "Gregory Smith", 
   "Anderson Smith",
   "Vincent Patternson", 
   "Keith Baron",
   "Derek Thomas Smith"]
)
word_counts = list_words.count()
print("Number of words in Resilient Distributed Data -> %i" % (word_counts))

collection_words = list_words.collect()
print("Elements in RDD -> %s" % (collection_words))

filtered_words = list_words.filter(lambda y: 'Smith' in y)
filtered_collection = filtered_words.collect()
print("Fitered RDD -> %s" % (filtered_collection))

index = 1
list_words_map = list_words.map(lambda y: (y,index))
list_word_mapping = list_words_map.collect()
print("Key value pair -> %s" % (list_word_mapping))