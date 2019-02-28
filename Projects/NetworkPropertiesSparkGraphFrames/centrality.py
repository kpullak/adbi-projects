from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions
from graphframes import *
from pyspark.sql.functions import *

sc=SparkContext("local", "degree.py")
sqlContext = SQLContext(sc)

def closeness(g):
	
	# Get list of vertices. We'll generate all the shortest paths at
	# once using this list.
	print g.vertices
	lm = map(lambda row: row.id, g.vertices.collect())
	print lm
	# first get all the path lengths.
	results = g.shortestPaths(landmarks = lm)
	results.select("id", "distances").show()
	# Break up the map and group by ID for summing
	shortPathLens = results.select("id", explode("distances"))
	shortPathLens.show()
	
	# Sum by ID
	centarlityMatrix = shortPathLens.groupBy(shortPathLens.id).agg(sum("value").alias("closeness"))

	# Get the inverses and generate desired dataframe.
	centarlityMatrix =  centarlityMatrix.map(lambda row: (row.id,1/float(row.closeness))).toDF().select(col("_1").alias("id"), col("_2").alias("closeness"))
	return centarlityMatrix

print("Reading in graph for problem 2.")
graph = sc.parallelize([('A','B'),('A','C'),('A','D'),
	('B','A'),('B','C'),('B','D'),('B','E'),
	('C','A'),('C','B'),('C','D'),('C','F'),('C','H'),
	('D','A'),('D','B'),('D','C'),('D','E'),('D','F'),('D','G'),
	('E','B'),('E','D'),('E','F'),('E','G'),
	('F','C'),('F','D'),('F','E'),('F','G'),('F','H'),
	('G','D'),('G','E'),('G','F'),
	('H','C'),('H','F'),('H','I'),
	('I','H'),('I','J'),
	('J','I')])
	
e = sqlContext.createDataFrame(graph,['src','dst'])
v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()
print("Generating GraphFrame.")
g = GraphFrame(v,e)

print("Calculating closeness.")
centrality = closeness(g).sort('closeness',ascending=False)
centrality.show()

print("Writing distribution to file centrality_out.csv")
centrality.toPandas().to_csv("centrality_out.csv")

