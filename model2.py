# importing required libraries

import json
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext

from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.sql.functions import length

from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np

# getting the sparkContext - which creates a new spark job

sc = SparkContext("local[2]", "DetectionOfSpam")
ssc = StreamingContext(sc, 1)
spark = SparkSession.builder.getOrCreate()


#Naive Bayes Bernoulli Classifier
def model1(data):
	data_array =  np.array(data.select("features").collect())
	label_data=np.array(data.select("label").collect())
	nsamples, nx, ny = data_array.shape
	data_array = data_array.reshape((nsamples,nx*ny))
	X_train, X_test, y_train, y_test = train_test_split(data_array,label_data,test_size=0.33, random_state=42)
	clf = BernoulliNB()
	
	clf.partial_fit(X_train, y_train,classes=np.unique(y_train))
	y_pred=clf.predict(X_test)
	print(clf.score(X_test,y_test))
	print(confusion_matrix(y_test,y_pred))

        
        
def tokenizer(df):
	tokenizer = Tokenizer(inputCol="feature1", outputCol="token_text")
	df=tokenizer.transform(df)
	return df

def stopremove(df):
	stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
	df=stopremove.transform(df)
	return df

def countvec(df):
	count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
	model=count_vec.fit(df)
	df=model.transform(df)
	return df

def idf(df):
	idf = IDF(inputCol="c_vec", outputCol="tf_idf")
	model=idf.fit(df)
	df=model.transform(df)
	return df

def stringindexer(df):
	ham_spam_to_num = StringIndexer(inputCol='feature2',outputCol='label')
	model=ham_spam_to_num.fit(df)
	df=model.transform(df)
	return df

def cleanup(df):
	clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')
	df=clean_up.transform(df)
	return df
	
	
	

def dataclean(df):
	df = df.withColumn('length',length(df['feature1']))
	df1=tokenizer(df)
	df2=stopremove(df1)
	df3=countvec(df2)
	df4=idf(df3)
	df5=stringindexer(df4)
	cleandata=cleanup(df5)
	return cleandata;
	
	
	
# Function to read the data stream and print the respective data read.

def RDDtoDf(x):
    if not x.isEmpty():
        y = x.collect()[0]
        z = json.loads(y)
        df=spark.createDataFrame(z.values())

        cleandata=dataclean(df)
      
        model1(cleandata)
 

records = ssc.socketTextStream("localhost", 6100)
records.foreachRDD(RDDtoDf)

ssc.start()
ssc.awaitTermination()
ssc.stop()
