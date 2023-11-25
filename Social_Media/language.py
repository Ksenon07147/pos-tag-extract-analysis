#!/usr/bin/env python
# coding: utf-8

# ## Language Detection

# In[1]:


from pyspark.sql import SparkSession
from pyspark import StorageLevel
import os

os.environ["PYSPARK_PYTHON"]="/home/pc/g5_env/bin/python39"

spark = SparkSession.builder.master("local[*]")\
            .appName("Language Detection")\
            .config('spark.executor.memory', '30g')\
            .config('spark.driver.maxResultSize', '30g')\
            .config('spark.driver.memory', '30g')\
            .getOrCreate()
            # .config('spark.ui.showConsoleProgress', False)\
           

sparkContext = spark.sparkContext

spark


# In[2]:


file_path = "hdfs://g5.bigtop.it:8020/user/root/filtered_social_media.parquet/part-00000-44f708f2-eb56-4495-8536-dd03b0326bbc-c000.snappy.parquet"


# In[3]:


df1 = spark.read.option("header",True).parquet(file_path)
df1.show()


# In[4]:


df1.rdd.getNumPartitions()


# In[5]:


df1 = df1.repartition(110)


# In[6]:


df1.rdd.getNumPartitions()


# In[7]:


df1.show()


# In[8]:


from pyspark.sql.functions import col

gram2 = df1.select(col("2grams")).withColumnRenamed("2grams","n-grams")
gram3 = df1.select(col("3grams")).filter(df1.word_count > 2).withColumnRenamed("3grams","n-grams")
gram4 = df1.select(col("4grams")).filter(df1.word_count > 3).withColumnRenamed("4grams","n-grams")
gram5 = df1.select(col("5grams")).filter(df1.word_count > 4).withColumnRenamed("5grams","n-grams")


# In[9]:


n_gram = gram5.union(gram4).union(gram3).union(gram2).persist(StorageLevel.MEMORY_ONLY)


# In[10]:


n_gram.show(1, False)


# In[11]:


from pyspark.sql.functions import explode
ngrams = n_gram.select(explode(col('n-grams'))).withColumnRenamed("col","ngrams")


# In[12]:


ngrams.show()


# In[13]:


def f(x): return x   
def exchangePosition(x, y):
    return y, x


# In[14]:


ngrams2 = ngrams.rdd.map(lambda x: (1,x))\
.flatMapValues(f)\
.map(lambda x: exchangePosition(x[0], x[1]))\
.reduceByKey(lambda a,b: a+b)\
.persist(StorageLevel.MEMORY_ONLY)


# In[15]:


ngrams2.count()


# In[16]:


column = ['ngram', 'gram_count']
ngrams2 = ngrams2.toDF(column)


# In[17]:


ngrams2.show()


# In[18]:


ngram2 = ngrams2.drop('gram_count')


# In[19]:


ngram2.show()


# In[20]:


ngram2.printSchema()


# In[21]:


import pyspark.sql.functions as f
ngram2 = ngram2.withColumn('word_count', f.size(f.split(f.col('ngram'), ' '))).persist(StorageLevel.MEMORY_ONLY)
ngram2.show(10, False)


# In[22]:


ngram2.count()


# In[23]:


keywords =  ['quality', ' service', '购买', '卖家', 'kemas']

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


# In[24]:


from pyspark.sql import Row

row = Row("ngram")
ngram3 = ngram2.rdd.map(lambda x: (keyword_position(x[0], x[1]))).map(row).toDF().dropna(how='any').persist(StorageLevel.MEMORY_ONLY)


# In[25]:


ngram3.count()


# In[26]:


from pyspark.sql.types import StringType
from lingua import Language, LanguageDetectorBuilder
from pyspark.sql.functions import col 

def lang_detect_word(text):
    languages = [Language.ENGLISH, Language.MALAY, Language.CHINESE]
    detector = LanguageDetectorBuilder.from_languages(*languages)\
    .with_minimum_relative_distance(0.1)\
    .build()
    
    empList = []
    
    for x in text.split(' '):
        language = detector.detect_language_of(x)
        if language == None:
            empList.append('OOV')
        elif language.name == 'ENGLISH':
            empList.append('EN')
        elif language.name == 'MALAY':
            empList.append('MS')
        elif language.name == 'CHINESE':
            empList.append('ZH')    

    return empList

rdd_lang_detect = ngram3.rdd.map(lambda x:  (x[0],lang_detect_word(x[0]))).persist(StorageLevel.MEMORY_ONLY)
rdd_lang_detect.count()


# In[27]:


rdd_lang_detect.toDF().show(truncate = False)


# In[28]:


def lang_detect_sentence(text):
    languages = [Language.ENGLISH, Language.MALAY, Language.CHINESE]
    detector = LanguageDetectorBuilder.from_languages(*languages)\
    .build()
    
    language = detector.detect_language_of(text)
    if language == None:
        return 'None'
    elif language.name == 'ENGLISH':
        
        return 'EN'
    elif language.name == 'MALAY':
        return 'MS'
    elif language.name == 'CHINESE':
        return 'ZH'
    return 'None'


# In[29]:


lang_detect = rdd_lang_detect.map(lambda x: (x[0], x[1], lang_detect_sentence(x[0]))).persist(StorageLevel.MEMORY_ONLY)


# In[30]:


lang_detect.count()


# In[31]:


column = ['sentence', 'language_word', 'language_sentence']


# In[32]:


df_lang_detect = lang_detect.toDF(column)


# In[33]:


df_lang_detect.show()


# In[34]:


from pyspark.sql.functions import col, concat_ws
df_final = df_lang_detect.withColumn('language_word', concat_ws(', ', col('language_word')))


# In[35]:


df_final.show()

df_final = df_final.coalesce(1)

df_final.write.parquet("hdfs://g5.bigtop.it:8020/user/root/language_social_media")