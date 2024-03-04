from pyspark import SparkContext 
spark_context = SparkContext("local", "Pyspark Accumulator application") 
snumdata = spark_context.accumulator(10) 
def g(y): 
   global snumdata 
   snumdata+=y 
spark_rdd = spark_context.parallelize([70,80,90,10]) 
spark_rdd.foreach(g) 
final_accum_value = snumdata.value 
print("Spark Accumulated value of Function g -> %i" % (final_accum_value))