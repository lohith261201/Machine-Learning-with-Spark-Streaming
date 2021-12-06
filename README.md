# Machine-Learning-with-Spark-Streaming
- Team ID-BD_203_227_301.
- This repository is dedicated for the big data project component.
- The topic we have chosen is Machine Learning With Spark Streaming.
- We have chosen Spam Analysis as our sub topic.

# Team Members

1) [LOHITH SRINIVAS T](https://github.com/lohith261201) PES2UG19CS203 
2) [MEELA DEEPTI](https://github.com/Priya2410) PES2UG19CS227
3) [PRIYA MOHATA](https://github.com/deeptimeela) PES2UG19CS301 


# Steps for streaming data :
- Place all the files in a single folder 
- Use stream.py provided by the TA's 
- Execute the following ``` python3 stream.py -f spam -b 20```
- Where here option f specifes the datase and b specifies the batch size
- Simultaneously execute ```$SPARK_HOME/opt/bin/spark-submit <path to streaming.py>```

# For the preprocessing references used: 
-  https://spark.apache.org/docs/latest/ml-features
-  https://www.codespeedy.com/spam-classification-using-pyspark-in-python/

# Methodologies used for preprocessing :
- Tokenizer
- Stop Words Removal
- Count Vectorizer
- String Indexer - to convert spam to 1 and ham to 0
- Vector Assembler

# Output after preprocessing : 
![priya-bd-2](https://user-images.githubusercontent.com/56394628/144718670-49e91e19-b757-4be1-b550-83362f80145b.png)

# Models Used :
- Multinomial Naive Bayes Classifier
- SGD Classifier 
- Mini Batch K-Means
- Perceptron 

# Evaluation of the Model :
- Accuracy
- Confusion Matrix 

# References for multinomial naive bayes classifier-
- https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

# References for sgd linear regressor:
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html




