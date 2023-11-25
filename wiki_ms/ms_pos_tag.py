#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark import StorageLevel
import os

os.environ["PYSPARK_PYTHON"]="/home/pc/g5_env_tf/bin/python39"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

spark = SparkSession.builder.master("local[*]")\
            .appName("ms_pos_tagging")\
            .config('spark.executor.memory', '20g')\
            .config('spark.driver.maxResultSize', '10g')\
            .config('spark.driver.memory', '20g')\
            .config('spark.ui.showConsoleProgress', False)\
            .getOrCreate()

sparkContext = spark.sparkContext

spark


# In[2]:


file_path = 'hdfs://g5.bigtop.it:8020/user/root/filtered_ms_wiki.parquet/part-00000-aa4a2315-60e8-4435-bd76-cf008453f11e-c000.snappy.parquet'


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


n_gram.printSchema()


# In[8]:


from pyspark.sql.functions import explode
ngrams = n_gram.select(explode(col('n-grams'))).withColumnRenamed("col","ngrams")


# In[9]:


ngrams.show()


# In[10]:


def f(x): return x   
def exchangePosition(x, y):
    return y, x


# In[11]:


ngrams2 = ngrams.rdd.map(lambda x: (1,x))\
.flatMapValues(f)\
.map(lambda x: exchangePosition(x[0], x[1]))\
.reduceByKey(lambda a,b: a+b)\
.persist(StorageLevel.MEMORY_ONLY)


# In[12]:


ngrams2.count()


# In[13]:


column = ['ngram', 'gram_count']
ngrams2 = ngrams2.toDF(column)


# In[14]:


ngrams2.show()


# In[15]:


ngram2 = ngrams2.drop('gram_count')


# In[16]:


ngram2.show()


# In[17]:


import pyspark.sql.functions as f
ngram2 = ngram2.withColumn('word_count', f.size(f.split(f.col('ngram'), ' '))).persist(StorageLevel.MEMORY_ONLY)
ngram2.count()


# In[18]:


ngram2.show(10, False)


# In[19]:


keywords =  ['terletak', 'kawasan', 'malaysia',' tentera', 'daerah',  'kabupaten']

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


# In[20]:


from pyspark.sql import Row

row = Row("ngram")
ngram3 = ngram2.rdd.map(lambda x: (keyword_position(x[0], x[1]))).map(row).toDF().dropna(how='any').persist(StorageLevel.MEMORY_ONLY)


# In[21]:


ngram3.count()


# In[22]:


import malaya
import logging

def ms_pos_tag(text):
    # logging.basicConfig(level=logging.CRITICAL)
    results = malaya.pos.transformer('alxlnet', True).predict(text)
    
    empList = []

    for x, y in results:
        empList.append(y)
    return empList


# In[23]:


sampling_ngram = ngram3.limit(10000)


# In[24]:


sampling_ngram.show()


# In[25]:


sampling_ngram.rdd.getNumPartitions()


# In[26]:


sampling_ngram = sampling_ngram.repartition(48)


# In[27]:


sampling_ngram.count()


# In[28]:


pos_tag = sampling_ngram.rdd.map(lambda x: (x[0], ms_pos_tag(x[0]))).persist(StorageLevel.MEMORY_ONLY)


# In[29]:


pos_tag.count()


# In[30]:


columns = ['sentence', 'pos_tag']

df_pos_tag = pos_tag.toDF(columns)


# In[31]:


df_pos_tag.show()


# In[32]:


import pyspark.sql.functions as f
df_final = df_pos_tag.withColumn('n-gram', f.size(f.split(f.col('sentence'), ' ')))


# In[33]:


df_final.show()


# In[34]:


from pyspark.sql.functions import col, concat_ws
df_final = df_final.withColumn('pos_tag', concat_ws(', ', col('pos_tag')))


# In[35]:


df_final.show()


# In[36]:


df_final = df_final.coalesce(1)

df_final.write.csv('ms_wiki_final',header = 'true')


# In[37]:


df_final.write.parquet('hdfs://g5.bigtop.it:8020/user/root/ms_wiki_final')

