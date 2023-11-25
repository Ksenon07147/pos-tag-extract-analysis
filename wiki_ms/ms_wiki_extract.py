#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark import StorageLevel
import os

os.environ["PYSPARK_PYTHON"]="/home/pc/g5_env/bin/python39"

spark = SparkSession.builder.master("local[16]")\
            .appName("ms_wiki_extract")\
            .config('spark.executor.memory', '20g')\
            .config('spark.driver.maxResultSize', '10g')\
            .config('spark.driver.memory', '20g')\
            .config('spark.ui.showConsoleProgress', False)\
            .getOrCreate()

sparkContext = spark.sparkContext

spark


# In[2]:


file_path = 'hdfs://g5.bigtop.it:8020/user/root/wikidata_ms_preprocessed.parquet/part-00000-a8275472-ea44-4f76-b4d6-78236b446b1d-c000.snappy.parquet'


# In[3]:


df1 = spark.read.option("header",True).parquet(file_path)
df1.show()


# In[4]:


df1.rdd.getNumPartitions()


# In[5]:


df1 = df1.repartition(64)


# In[6]:


df1.show()


# In[7]:


from pyspark.sql.functions import col,isnan, when, count

c = '_1'
df1.select([count(when(col(c).isNull(), c)).alias(c)]).show()


# In[8]:


from nltk.tokenize import word_tokenize
import re
def word_token(text):
    string =text['_1']
    # print(type(text))
    return word_tokenize(string)

def f(x): return x

def exchangePosition(text, index):
    return index, (text)


# In[9]:


RDD = df1.rdd.map(lambda x: word_token(x))\
.persist(StorageLevel.MEMORY_ONLY)


# In[10]:


RDD2=RDD.map(lambda x:(x,1)).map(lambda x: exchangePosition(x[0], x[1]))\
.flatMapValues(f)\
.map(lambda x: exchangePosition(x[0], x[1]))\
.reduceByKey(lambda a,b: a+b)\
.persist(StorageLevel.MEMORY_ONLY)


# In[11]:


RDD2.count()


# In[12]:


Column = ['word', 'word_count']
df2 = RDD2.toDF(Column)


# In[13]:


df2.show()


# In[14]:


import advertools as adv
indo_STOPWORDS=adv.stopwords['indonesian']


# In[15]:


df2.orderBy("word_count", ascending=False)\
.filter(~col("word").isin(indo_STOPWORDS))\
.show(10)


# In[16]:


def text_extract(text):
    keywords =  ['terletak', 'kawasan', 'malaysia',' tentera', 'daerah',  'kabupaten']
    for x in keywords:
        if x in text:
            return text
        
    return ''


keyword_filter = RDD.map(lambda x: text_extract(x)).filter(lambda x: x != '').persist(StorageLevel.MEMORY_ONLY)


# In[17]:


keyword_filter.count()


# In[18]:


data1 = keyword_filter.map(lambda x : (tuple(x), 1))\
.reduceByKey(lambda a,b: a+b).persist(StorageLevel.MEMORY_ONLY)


# In[19]:


from pyspark.sql.types import StructType,StructField, StringType, IntegerType, ArrayType
schema = StructType([ \
    StructField('token_sentence',ArrayType(StringType()),True), \
    StructField('sent_count',IntegerType(),True), \
  ])


df1 = spark.createDataFrame(data = data1, schema=schema)
df1.show()


# In[20]:


import pyspark.sql.functions as f
df1 = df1.withColumn('word_count', f.size(f.col('token_sentence')))


# In[21]:


df1.show()


# In[22]:


df1.filter(df1.word_count == 1).show()


# In[23]:


df2 = df1.filter(df1.word_count > 1)


# In[24]:


df2.show()


# In[25]:


df2.count()


# ### Ngram

# In[26]:


from pyspark.ml.feature import NGram
ngramDataFrame = df2

for x in range(2,6):
    ngram = NGram(n=x, inputCol='token_sentence', outputCol= str(x) +'grams')
    ngramDataFrame = ngram.transform(ngramDataFrame) 
ngramDataFrame.show(1)


# In[27]:


ngramDataFrame.filter(ngramDataFrame.word_count < 5).count()


# In[28]:


ngramDataFrame.filter(ngramDataFrame.word_count == 4).show()


# In[29]:


ngramDataFrame.filter(ngramDataFrame.word_count == 3).show()


# In[30]:


ngramDataFrame.filter(ngramDataFrame.word_count == 2).show()


# In[31]:


ngramDataFrame.filter(ngramDataFrame.word_count == 2)

def find_keyword(text):
    keywords =  ['terletak', 'kawasan', 'malaysia',' tentera', 'daerah',  'kabupaten']
    
    empList = []
    for x in text:
        for y in keywords:
            if y in x:
                empList.append(x)
    return empList


# In[32]:


column =  ['token_sentence','sent_count','word_count','2grams','3grams','4grams','5grams']
gram5= ngramDataFrame.filter(ngramDataFrame.word_count > 4)\
.rdd.map(lambda x: (x[0], x[1], x[2],find_keyword(x[3])\
,find_keyword(x[4]), find_keyword(x[5]), find_keyword(x[6]))).persist(StorageLevel.MEMORY_ONLY)


# In[33]:


df_5gram = gram5.toDF(column).show()


# In[34]:


gram4= ngramDataFrame.filter(ngramDataFrame.word_count == 4)\
.rdd.map(lambda x: (x[0], x[1], x[2],find_keyword(x[3])\
,find_keyword(x[4]), find_keyword(x[5]), x[6])).persist(StorageLevel.MEMORY_ONLY)


# In[35]:


gram3= ngramDataFrame.filter(ngramDataFrame.word_count == 3)\
.rdd.map(lambda x: (x[0], x[1], x[2],find_keyword(x[3])\
,find_keyword(x[4]), x[5], x[6])).persist(StorageLevel.MEMORY_ONLY)


# In[36]:


gram2= ngramDataFrame.filter(ngramDataFrame.word_count == 2)\
.rdd.map(lambda x: (x[0], x[1], x[2],find_keyword(x[3])\
,find_keyword(x[4]), x[5], x[6])).persist(StorageLevel.MEMORY_ONLY)


# In[37]:


Final_gram = gram5.union(gram4).union(gram3).union(gram2).toDF(column).persist(StorageLevel.MEMORY_ONLY)


# In[38]:


Final_gram.show()


# In[39]:


Final_gram.count()


# In[40]:


Final_gram.filter(Final_gram.word_count == 4).show()


# In[41]:


Final_gram = Final_gram.coalesce(1)

Final_gram.write.parquet("hdfs://g5.bigtop.it:8020/user/root/filtered_ms_wiki.parquet")

