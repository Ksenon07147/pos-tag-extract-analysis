#!/usr/bin/env python
# coding: utf-8

# In[1]:


# PySpark
from pyspark.sql import SparkSession
import os

os.environ["PYSPARK_PYTHON"]="/home/pc/g5_env/bin/python39"


# In[2]:


# Schema declaration
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
#map reduce i think
from functools import reduce
#pyspark sql functions
from pyspark.sql import functions as f
# storage level
from pyspark import StorageLevel


# In[3]:


spark = SparkSession.builder.master("local[12]")\
            .appName("Read CSV")\
            .config('spark.executor.memory', '10g')\
            .config('spark.driver.maxResultSize', '5g')\
            .config('spark.driver.memory', '10g')\
            .getOrCreate()
sparkContext = spark.sparkContext
spark


# In[4]:


spark.conf.get('spark.sql.files.maxPartitionBytes')


# In[5]:


spark.conf.set('spark.sql.files.maxPartitionBytes', '256000000b')
spark.conf.get('spark.sql.files.maxPartitionBytes')

# run when idling
spark.stop()
# In[6]:


df = spark.read.option("wholetext",False)\
    .text("/home/pc/ozj/wiki-data-ms-csv-test-remove-html.txt")


# In[7]:


df.show(3,truncate=False)


# In[8]:


df = df.dropna(how='any').where(df.value != ' ')


# In[9]:


df.rdd.getNumPartitions()


# In[10]:


df.rdd.getNumPartitions()


# In[11]:


df = df.repartition(48)


# In[12]:


df.rdd.getNumPartitions()


# ### <font color='red'>Sentence Tokenize</font>

# In[13]:


from nltk.tokenize import sent_tokenize, word_tokenize
def sentence_tokenize(text, index):
    return index,sent_tokenize(text)

def f(x): return x


# In[14]:


from pyspark.sql.functions import col, explode, regexp_replace, length

new_df = df.select("value") \
    .where(length(col("value")) >= 900)
new_df.count()


# In[15]:


df.count()


# In[16]:


# Adding indexes to all sentence
# Swapping the column
# Splitting the sentence
RDD1 = df.rdd.map(lambda x: x[0]).zipWithIndex()\
.map(lambda x: sentence_tokenize(x[0], x[1]))\
.flatMapValues(f)\
.persist(StorageLevel.MEMORY_ONLY)
# .toDF().show(100, truncate = False)


# Remove html code and header (== like this ==)

# ## <font color='red'>Text Cleaning</font>
# 

# #### Remove HTML tag

# In[17]:


def remove_html_tags(text):
    #Remove html tags from a string
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', text)


# #### Remove contractions

# In[18]:


import contractions

def contractions_word(text):
    # using contractions.fix to expand the shortened words
    try:
        text = contractions.fix(text)
    except:
        return text
    return text
# RDD = dataDF.rdd.map(lambda x: x[0]).zipWithIndex()\
# .map(lambda x: sentence_tokenize(x[0], x[1]))\
# .flatMapValues(f)\
# .map(lambda x : remove_html_tags(x[1]))\
# .map(lambda x : contractions_word(x))\
# .map(lambda x: remove_emoticons(x))\
# .map(lambda x : sent_tokenize(x))\
# .persist(StorageLevel.MEMORY_ONLY)


# #### Remove Emoticons

# In[19]:


from emot.emo_unicode import EMOTICONS_EMO

def remove_emoticons(text):
    for emot in EMOTICONS_EMO:
        text = text.replace(emot, '')
    return text

#RDD2 = RDD.map(lambda x: remove_emoticons(x)).persist(StorageLevel.MEMORY_ONLY)


# In[20]:


RDD2 = RDD1.map(lambda x : remove_html_tags(x[1]))\
.map(lambda x : contractions_word(x))\
.map(lambda x: remove_emoticons(x))\
.persist(StorageLevel.MEMORY_ONLY)
# .map(lambda x : sent_tokenize(x))\


# In[21]:


RDD3 = RDD2.zipWithIndex()\
.map(lambda x: sentence_tokenize(x[0], x[1]))\
.flatMapValues(f)\
.persist(StorageLevel.MEMORY_ONLY)

RDD3.toDF().show(100,truncate=False)
# #### 1) Remove Emoji, URL, Phone Number, Currency, Punctuations, Digit, E-mail
# #### 2) To Lower

# In[22]:


from cleantext import clean

def clean_text(text): 
    return str(clean(text,
            fix_unicode=True,               # fix various unicode errors
            to_ascii=False,                  # transliterate to closest ASCII representation
            lower=True,                     # lowercase text
            no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
            no_urls=True,                  # replace all URLs with a special token
            no_emails=True,                # replace all email addresses with a special token
            no_phone_numbers=True,         # replace all phone numbers with a special token
            no_numbers=True,               # replace all numbers with a special token
            no_digits=False,                # replace all digits with a special token
            no_currency_symbols=True,      # replace all currency symbols with a special token
            no_punct=True,                 # remove punctuations
            no_emoji=True,
            replace_with_punct="",          # instead of removing punctuations you may replace them
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="",
            replace_with_number="",
            replace_with_digit="",
            replace_with_currency_symbol="",
            lang="en"                       # set to 'de' for German special handling
            ))

# RDD2 = RDD.map(lambda x: remove_emoticons(x))\
# .map(lambda x : (clean_text(x)))\
# .filter(lambda x : x != '')\
# .persist(StorageLevel.MEMORY_ONLY)


# #### Remove Symbols

# In[23]:


def remove_symbols(text):
    symbols = ['+', '^', '|', '~', '>', '<', '=', '`']
    for x in symbols:
        text = text.replace(x, ' ')
    return text


# In[24]:


RDD4 = RDD3.map(lambda x : (clean_text(x)))\
.filter(lambda x : x != '')\
.map(lambda x : (remove_symbols(x)))\
.filter(lambda x : x != '')\
.persist(StorageLevel.MEMORY_ONLY)


# In[25]:


df2 = RDD4.map(lambda x: (x, )).toDF()


# In[26]:


from pyspark.sql.functions import col, explode, regexp_replace, length

new_df = df2.select("_1")\
    .where(col("_1").isNotNull()) \

#.where(length(col("_1")) >= 900)

new_df.show(1000,truncate=False)


# In[28]:


import re

final_df = new_df.rdd.map(lambda x: (re.sub(r'[^a-zA-Z0-9 ]', '', x[0]), ))\
.map(lambda x: (re.sub(' +', ' ', x[0]), ))\
.toDF()


# In[29]:


final_df.show(200,truncate=False)


# ##### Save Result to HDFS

# In[30]:


final_df = final_df.coalesce(1)


# In[32]:


final_df.write.parquet("hdfs://g5.bigtop.it:8020/user/root/wikidata_ms_preprocessed.parquet") 


# In[33]:


df_test_par = spark.read.parquet("hdfs://g5.bigtop.it:8020/user/root/wikidata_ms_preprocessed.parquet")


# In[34]:


df_test_par.show(60,truncate=False)

