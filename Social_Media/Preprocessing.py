#!/usr/bin/env python
# coding: utf-8

# # <font color='red'>Social Media Data Preproccessing</font> 

# ## <font color='red'>Start Spark Session</font> 

# In[1]:


from pyspark.sql import SparkSession
from pyspark import StorageLevel
import os

os.environ["PYSPARK_PYTHON"]="/home/pc/g5_env/bin/python39"

spark = SparkSession.builder.master("local[22]")\
            .appName("preproccesing")\
            .config('spark.executor.memory', '20g')\
            .config('spark.driver.maxResultSize', '10g')\
            .config('spark.driver.memory', '20g')\
            .config('spark.ui.showConsoleProgress', False)\
            .getOrCreate()

sparkContext = spark.sparkContext

spark


# In[2]:


spark.conf.get('spark.sql.files.maxPartitionBytes')


# In[3]:


spark.conf.set('spark.sql.files.maxPartitionBytes', 64000000)
spark.conf.get('spark.sql.files.maxPartitionBytes')


# ## <font color='red'>Import Social Media Data</font> 

# In[4]:


dataDF = spark.read.option("header",True).csv("/home/pc/data/parsed_data/4data-comment_only.csv")
dataDF.show(10)


# In[5]:


dataDF.rdd.getNumPartitions()


# In[6]:


dataDF = dataDF.repartition(66)


# In[7]:


dataDF.rdd.getNumPartitions()


# ## <font color='red'>Dealing With Null Value</font>

# In[8]:


### Checking total number of null value
from pyspark.sql.functions import col,isnan, when, count

c = 'Comment'
dataDF.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c)]).show()


# In[9]:


### Remove null value

dataDF = dataDF.dropna(how='any')


# ### <font color='red'>Sentence Tokenize</font>

# In[10]:


from nltk.tokenize import sent_tokenize, word_tokenize
def sentence_tokenize(text, index):
    return index,sent_tokenize(text)

def f(x): return x


# In[11]:


# Adding indexes to all sentence
# Swapping the column
# Splitting the sentence
RDD = dataDF.rdd.map(lambda x: x[0]).zipWithIndex()\
.map(lambda x: sentence_tokenize(x[0], x[1]))\
.flatMapValues(f)\
.persist(StorageLevel.MEMORY_ONLY)


# ## <font color='red'>Text Cleaning</font>
# 

# #### Remove HTML tag

# In[12]:


def remove_html_tags(text):
    #Remove html tags from a string
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', text)


# #### Remove contractions

# In[13]:


import contractions

def contractions_word(text):
    # using contractions.fix to expand the shortened words
    try:
        text = contractions.fix(text)
    except:
        return text
    return text


# #### Remove Emoticons

# In[14]:


from emot.emo_unicode import EMOTICONS_EMO

def remove_emoticons(text):
    for emot in EMOTICONS_EMO:
        text = text.replace(emot, '')
    return text


# In[15]:


RDD2 = RDD.map(lambda x : remove_html_tags(x[1]))\
.map(lambda x : contractions_word(x))\
.map(lambda x: remove_emoticons(x))\
.persist(StorageLevel.MEMORY_ONLY)



# In[16]:


RDD3 = RDD2.zipWithIndex()\
.map(lambda x: sentence_tokenize(x[0], x[1]))\
.flatMapValues(f)\
.persist(StorageLevel.MEMORY_ONLY)


# #### 1) Remove Emoji, URL, Phone Number, Currency, Punctuations, Digit, E-mail
# #### 2) To Lower

# In[17]:


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
            no_digits=True,                # replace all digits with a special token
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


# #### Remove Symbols

# In[18]:


def remove_symbols(text):
    symbols = ['+', '^', '|', '~', '>', '<', '=', '`']
    for x in symbols:
        text = text.replace(x, ' ')
    return text


# In[19]:


RDD4 = RDD3.map(lambda x : (clean_text(x)))\
.filter(lambda x : x != '')\
.map(lambda x : (remove_symbols(x)))\
.filter(lambda x : x != '')\
.persist(StorageLevel.MEMORY_ONLY)

column = ['sentence']
df_clean = RDD4.map(lambda x: (x,)).toDF(column)df_clean.show(truncate = False)df_clean = df_clean.coalesce(1)

df_clean.write.parquet("hdfs://g5.bigtop.it:8020/user/root/preprocess_social_media.parquet")
# ## <font color='red'>Word Tokenize</font>

# In[20]:


import jieba
import logging

def my_jieba(text):
    jieba.setLogLevel(logging.WARNING)
    return jieba.lcut(text, cut_all=False)

def remove_space(text):
    empList = []

    for x in text: 
        if x != '':
            if x != ' ':
                empList.append(x)
    return empList

def exchangePosition(text, index):
    return index, (text)


# In[21]:


RDD5 = RDD4.map(lambda x: my_jieba(x))\
.map(lambda x: remove_space(x)).persist(StorageLevel.MEMORY_ONLY)


# In[22]:


#Possible Ngram final output
data = RDD5.map(lambda x: (x,1))\
.map(lambda x: exchangePosition(x[0], x[1]))\
.flatMapValues(f)\
.map(lambda x: exchangePosition(x[0], x[1]))\
.reduceByKey(lambda a,b: a+b)


# In[23]:


from pyspark.sql.types import StructType,StructField, StringType, IntegerType

schema = StructType([ \
    StructField('word',StringType(),True), \
    StructField('word_count',IntegerType(),True), \
  ])


df = spark.createDataFrame(data = data, schema=schema)


# In[24]:


df.show()


# In[25]:


import advertools as adv
from nltk.corpus import stopwords as stop_words

zh_STOPWORDS, indo_STOPWORDS, eng_STOPWORDS = adv.stopwords['chinese'], adv.stopwords['indonesian'], stop_words.words('english')


# In[26]:


df.orderBy("word_count", ascending=False)\
.filter(~col("word").isin(eng_STOPWORDS))\
.filter(~col("word").isin(zh_STOPWORDS))\
.filter(~col("word").isin(indo_STOPWORDS))\
.show(10)


# ## <font color='red'>Data Extraction</font> 

# In[27]:


def text_extract(text):
    keywords =  ['quality', ' service', '购买', '卖家', 'kemas']

    for x in keywords:
        if x in text:
            return text
        
    return ''


keyword_filter = RDD5.map(lambda x: text_extract(x)).filter(lambda x: x != '').persist(StorageLevel.MEMORY_ONLY)


# In[28]:


keyword_filter.count()


# In[29]:


data1 = keyword_filter.map(lambda x : (tuple(x), 1))\
.reduceByKey(lambda a,b: a+b).persist(StorageLevel.MEMORY_ONLY)


# In[30]:


from pyspark.sql.types import StructType,StructField, StringType, IntegerType, ArrayType
schema = StructType([ \
    StructField('token_sentence',ArrayType(StringType()),True), \
    StructField('sent_count',IntegerType(),True), \
  ])


df1 = spark.createDataFrame(data = data1, schema=schema)
df1.show()


# In[31]:


import pyspark.sql.functions as f
df1 = df1.withColumn('word_count', f.size(f.col('token_sentence')))


# In[32]:


df1.show()


# In[33]:


df1.filter(df1.word_count == 1).show()


# In[34]:


df2 = df1.filter(df1.word_count > 1)


# In[35]:


df2.count()


# ### NGRAM

# In[36]:


from pyspark.ml.feature import NGram
ngramDataFrame = df2

for x in range(2,6):
    ngram = NGram(n=x, inputCol='token_sentence', outputCol= str(x) +'grams')
    ngramDataFrame = ngram.transform(ngramDataFrame) 
ngramDataFrame.show()


# In[37]:


ngramDataFrame.filter(ngramDataFrame.word_count < 5).count()


# In[38]:


ngramDataFrame.filter(ngramDataFrame.word_count == 4).show()


# In[39]:


ngramDataFrame.filter(ngramDataFrame.word_count == 3).show()


# In[40]:


ngramDataFrame.filter(ngramDataFrame.word_count == 2).show()


# In[41]:


def find_keyword(text):
    keywords =  ['quality', ' service', '购买', '卖家', 'kemas']
    
    empList = []
    for x in text:
        for y in keywords:
            if y in x:
                empList.append(x)
    return empList


# In[42]:


column =  ['token_sentence','sent_count','word_count','2grams','3grams','4grams','5grams']
gram5= ngramDataFrame.filter(ngramDataFrame.word_count > 4)\
.rdd.map(lambda x: (x[0], x[1], x[2],find_keyword(x[3])\
,find_keyword(x[4]), find_keyword(x[5]), find_keyword(x[6]))).persist(StorageLevel.MEMORY_ONLY)


# In[43]:


df_5gram = gram5.toDF(column).show()


# In[44]:


gram4= ngramDataFrame.filter(ngramDataFrame.word_count == 4)\
.rdd.map(lambda x: (x[0], x[1], x[2],find_keyword(x[3])\
,find_keyword(x[4]), find_keyword(x[5]), x[6])).persist(StorageLevel.MEMORY_ONLY)


# In[45]:


gram3= ngramDataFrame.filter(ngramDataFrame.word_count == 3)\
.rdd.map(lambda x: (x[0], x[1], x[2],find_keyword(x[3])\
,find_keyword(x[4]), x[5], x[6])).persist(StorageLevel.MEMORY_ONLY)


# In[46]:


gram2= ngramDataFrame.filter(ngramDataFrame.word_count == 2)\
.rdd.map(lambda x: (x[0], x[1], x[2],find_keyword(x[3])\
,find_keyword(x[4]), x[5], x[6])).persist(StorageLevel.MEMORY_ONLY)


# In[47]:


Final_gram = gram5.union(gram4).union(gram3).union(gram2).toDF(column).persist(StorageLevel.MEMORY_ONLY)


# In[48]:


Final_gram.show()


# In[49]:


Final_gram.count()


# In[51]:


Final_gram.filter(Final_gram.word_count > 4).show()

Final_gram = Final_gram.coalesce(1)

# ### Change Format
from pyspark.sql.functions import col, concat_ws
df_final = df2.withColumn('token_sentence', concat_ws(', ', col('token_sentence')))
df_final.show()
df_final.orderBy("sent_count","word_count", ascending=False).show()
df_final.count()
# ### To CSV
df_final = df_final.coalesce(1)

df_final.write.format('com.databricks.spark.csv').save('filter',header = 'true')