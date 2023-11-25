#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark import StorageLevel
import os

os.environ["PYSPARK_PYTHON"]="/home/pc/g5_env/bin/python39"

spark = SparkSession.builder.master("local[5]")\
            .appName("en_pos_tag")\
            .config('spark.executor.memory', '20g')\
            .config('spark.driver.maxResultSize', '10g')\
            .config('spark.driver.memory', '20g')\
            .config('spark.ui.showConsoleProgress', False)\
            .getOrCreate()

sparkContext = spark.sparkContext

spark


# In[2]:


file_path = 'hdfs://g5.bigtop.it:8020/user/root/filtered_en_wiki.parquet/part-00000-b7931c44-7a8c-4f89-8265-6d16a044997a-c000.snappy.parquet'


# In[3]:


df1 = spark.read.option("header",True).parquet(file_path)
df1.show()


# In[4]:


from pyspark.sql.functions import col

gram2 = df1.select(col("2grams")).withColumnRenamed("2grams","n-grams")
gram3 = df1.select(col("3grams")).filter(df1.word_count > 2).withColumnRenamed("3grams","n-grams")
gram4 = df1.select(col("4grams")).filter(df1.word_count > 3).withColumnRenamed("4grams","n-grams")
gram5 = df1.select(col("5grams")).filter(df1.word_count > 4).withColumnRenamed("5grams","n-grams")


# In[5]:


n_gram = gram5.union(gram4).union(gram3).union(gram2).persist(StorageLevel.MEMORY_ONLY)


# In[6]:


n_gram.show()


# In[7]:


from pyspark.sql.functions import explode
ngrams = n_gram.select(explode(col('n-grams'))).withColumnRenamed("col","ngrams")


# In[8]:


ngrams.show()


# In[9]:


def f(x): return x   
def exchangePosition(x, y):
    return y, x


# In[10]:


ngrams2 = ngrams.rdd.map(lambda x: (1,x))\
.flatMapValues(f)\
.map(lambda x: exchangePosition(x[0], x[1]))\
.reduceByKey(lambda a,b: a+b)\
.persist(StorageLevel.MEMORY_ONLY)


# In[32]:


ngrams2.count()


# In[12]:


column = ['ngram', 'gram_count']
ngrams2 = ngrams2.toDF(column)


# In[13]:


ngrams2.show()


# In[14]:


ngram2 = ngrams2.drop('gram_count')


# In[15]:


ngram2.show()


# In[16]:


import pyspark.sql.functions as f
ngram2 = ngram2.withColumn('word_count', f.size(f.split(f.col('ngram'), ' '))).persist(StorageLevel.MEMORY_ONLY)
ngram2.count()


# In[17]:


ngram2.show(10, False)


# In[18]:


keywords =  ['secondary', 'school', 'tertiary', 'university', 'national',  'private']

def keyword_position(text, n_gram):

    keyword = []
    index_of_keyword =[]
    
    tempList = list(text.split(" "))
    
    for x in keywords:
        i = 0
        for y in tempList:
            if x == y:
                keyword.append(x)
                index_of_keyword.append(i)
            i = i + 1
    
    if keyword == []:
        return
    
    if n_gram == 3 or n_gram == 2:
        return text
    
    if n_gram == 5:
        if 2 in index_of_keyword:
            return text
    
    if n_gram == 4:
        if 1 in index_of_keyword or 2 in index_of_keyword:
            return text

    return


# In[19]:


from pyspark.sql import Row

row = Row("ngram")
ngram3 = ngram2.rdd.map(lambda x: (keyword_position(x[0], x[1]))).map(row).toDF().dropna(how='any').persist(StorageLevel.MEMORY_ONLY)


# In[20]:


from nltk import pos_tag, word_tokenize

def en_pos_tag(text):
    results = pos_tag(word_tokenize(text), lang='eng') 
    empList = []
    for x,y in results:
        empList.append(y)

    return empList


# In[21]:


ngram3.rdd.getNumPartitions()


# In[22]:


ngram3 = ngram3.repartition(48)


# In[23]:


ngram3.count()


# In[24]:


pos_tag = ngram3.rdd.map(lambda x: (x[0], en_pos_tag(x[0]))).persist(StorageLevel.MEMORY_ONLY)


# In[25]:


pos_tag.count()


# In[26]:


columns = ['sentence', 'pos_tag']

df_pos_tag = pos_tag.toDF(columns)


# In[27]:


df_pos_tag.show()


# In[28]:


import pyspark.sql.functions as f
df_final = df_pos_tag.withColumn('n-gram', f.size(f.split(f.col('sentence'), ' ')))


# In[29]:


df_final.show()


# In[30]:


from pyspark.sql.functions import col, concat_ws
df_final = df_final.withColumn('pos_tag', concat_ws(', ', col('pos_tag')))


# In[31]:


df_final.show()

df_final = df_final.coalesce(1)

df_final.write.csv('en_wiki_final',header = 'true')df_final.write.parquet('hdfs://g5.bigtop.it:8020/user/root/en_wiki_final')import pandas as pd
df = pd.read_csv("/home/pc/zerone/assignment/wiki/en_wiki_final/part-00000-6d13cf29-99e5-4ed6-9c8a-242a642a8e08-c000.csv",header=0)
df
# In[ ]:




