from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql import SQLContext

spark_context = SparkContext("local", "SQL Context")

sqlContext = SQLContext(spark_context)

data=[('Handbag',1),('Gold Chain',2),('Choco Bar',3),('Thai Curry Powder',4)] 

rdd = spark_context.parallelize(data)  


product_map=rdd.map(lambda y: Row(product=y[0], product_id=int(y[1])))            

data_frame = sqlContext.createDataFrame(product_map).collect()

print(data_frame)

