#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark import StorageLevel
import os

os.environ["PYSPARK_PYTHON"]="/home/pc/g5_env_tf/bin/python39"

spark = SparkSession.builder.master("local[*]")\
            .appName("PosTagging")\
            .config('spark.executor.memory', '30g')\
            .config('spark.driver.maxResultSize', '30g')\
            .config('spark.driver.memory', '30g')\
            .getOrCreate()
            # .config('spark.ui.showConsoleProgress', False)\
           

sparkContext = spark.sparkContext

spark


# In[2]:


file_path = "hdfs://g5.bigtop.it:8020/user/root/language_social_media/part-00000-0fc8e338-2c87-4f9e-9db3-4476b0a96443-c000.snappy.parquet"


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

language_none = df1.select(col("sentence"), col("language_word")).filter(df1.language_sentence == 'None')
language_en = df1.select(col("sentence"), col("language_word")).filter(df1.language_sentence == 'EN')
language_ms = df1.select(col("sentence"), col("language_word")).filter(df1.language_sentence == 'MS')
language_zh = df1.select(col("sentence"), col("language_word")).filter(df1.language_sentence == 'ZH')


# In[9]:


language_none.show()


# In[10]:


language_none.count()


# In[11]:


language_en.show()


# In[12]:


language_ms.show()


# In[13]:


language_ms.count()


# In[14]:


language_zh.show()


# In[15]:


from nltk import pos_tag, word_tokenize
def zh_pos_tag(text):    
    import logging
    import jieba
    import jieba.posseg
        
    jieba.setLogLevel(logging.WARNING)

    text = text.replace(' ','')
    results= jieba.posseg.lcut(text)
    empList = []
    for x, y in results:
        empList.append(y)
    
    return empList


# In[16]:


def en_pos_tag(text):
    results = pos_tag(word_tokenize(text), lang='eng') 
    empList = []
    for x,y in results:
        empList.append(y)

    return empList


# In[17]:


import malaya
import logging

def ms_pos_tag(text):
    # logging.basicConfig(level=logging.CRITICAL)
    results = malaya.pos.transformer('alxlnet', True).predict(text)
    
    empList = []

    for x, y in results:
        empList.append(y)
    return empList


# In[18]:


def none_pos_tag(text):
    # logging.basicConfig(level=logging.CRITICAL)
    empList = []
    tempList = list(text.split(" "))
    for x in tempList:
        empList.append('XXX')
            
    return empList


# In[19]:


language_en = language_en.repartition(110)


# In[20]:


language_en.count()


# In[21]:


en_pos_tag = language_en.rdd.map(lambda x: (x[0], x[1], en_pos_tag(x[0]))).persist(StorageLevel.MEMORY_ONLY)


# In[22]:


en_pos_tag.count()


# In[23]:


language_zh = language_zh.repartition(110)


# In[24]:


language_zh.count()


# In[25]:


zh_pos_tag = language_zh.rdd.map(lambda x: (x[0], x[1], zh_pos_tag(x[0]))).persist(StorageLevel.MEMORY_ONLY)


# In[26]:


zh_pos_tag.count()


# In[27]:


language_none= language_none.repartition(110)


# In[28]:


language_none.count()


# In[29]:


none_pos_tag = language_none.rdd.map(lambda x: (x[0], x[1], none_pos_tag(x[0]))).persist(StorageLevel.MEMORY_ONLY)


# In[30]:


none_pos_tag.count()


# ### English, Chinese, None language Result

# In[31]:


en_pos_tag.toDF().show()


# In[32]:


zh_pos_tag.toDF().show()


# In[33]:


none_pos_tag.toDF().show(truncate = False)


# In[34]:


zh_tags = {'a': 'ADJ',
 'ad': 'ACC_WORD',
 'ag': 'ADJ',
 'al': 'ADJ',
 'an': 'NOUN',
 'b': 'ADJ',
 'bl': 'ADJ',
 'c': 'CONJ',
 'cc': 'CONJ',
 'd': 'ADV',
 'f': 'NOUN',
 'm': 'NUM',
 'mq': 'MQ',
 'n': 'NOUN',
 'ng': 'XXX',
 'nl': 'NOUN',
 'nr': 'NOUN',
 'nr1': 'NOUN',
 'nr2': 'NOUN',
 'nrf': 'NOUN',
 'nrj': 'NOUN',
 'ns': 'NOUN',
 'nsf': 'NOUN',
 'nt': 'NOUN',
 'nz': 'NOUN',
 'p': 'IN',
 'pba': 'IN',
 'pbei': 'IN',
 'qt': 'QT',
 'qv': 'QV',
 'r': 'PRON',
 'rg': 'PRON',
 'rr': 'PRON',
 'ry': 'PRON',
 'rys': 'PRON',
 'ryt': 'PRON',
 'ryv': 'PRON',
 'rz': 'PRON',
 'rzs': 'PRON',
 'rzv': 'PRON',
 's': 'NOUN',
 't': 'TIME',
 'tg': 'TIME',
 'ul': 'UL',
 'v': 'VERB',
 'vd': 'ADX',
 'vf': 'VERB',
 'vg': 'VERB',
 'vi': 'VERB',
 'vl': 'VERB',
 'vn': 'NOUN_VERB',
 'vshi': 'VERB',
 'vx': 'VERB',
 'vyou': 'VERB',
 'x': 'XXX',
 'z': 'STATE_WORD'}


# In[35]:


en_tags = {'CC': 'CCONJ',
 'CD': 'NOUN',
 'DT': 'DET',
 'EX': 'ADV',
 'FW': 'XXX',
 'IN': 'IN',
 'JJ': 'ADJ',
 'JJR': 'ADJ',
 'JJS': 'ADJ',
 'LS': 'XXX',
 'MD': 'VERB',
 'NN': 'NOUN',
 'NNP': 'PROPN',
 'NNPS': 'PROPN',
 'NNS': 'NOUN',
 'PDT': 'DET',
 'POS': 'POS',
 'PRP': 'PRON',
 'PRP$': 'PRON',
 'RB': 'ADV',
 'RBR': 'ADV',
 'RBS': 'ADV',
 'RP': 'PART',
 'TO': 'TO',
 'UH': 'UH',
 'VB': 'VERB',
 'VBD': 'VERB',
 'VBG': 'VERB',
 'VBN': 'VERB',
 'VBP': 'VERB',
 'VBZ': 'VERB',
 'WDT': 'SCONJ',
 'WP': 'SCONJ',
 'WP$': 'SCONJ',
 'WRB': 'SCONJ'}


# In[36]:


def standardizer(output_list, dictionary):
    empList = []
    
    for x in output_list:
        if x not in dictionary:
            empList.append('XXX')
            continue
        y = dictionary.get(x)
        empList.append(y)
    return empList


# In[37]:


standard_zh_pos_tag = zh_pos_tag.map(lambda x: (x[0], x[1], standardizer(x[2], zh_tags))).persist(StorageLevel.MEMORY_ONLY)


# In[38]:


standard_zh_pos_tag.count()


# In[39]:


standard_zh_pos_tag.toDF().show()


# In[40]:


standard_en_pos_tag = en_pos_tag.map(lambda x: (x[0], x[1], standardizer(x[2], en_tags))).persist(StorageLevel.MEMORY_ONLY)


# In[41]:


standard_en_pos_tag.count()


# In[42]:


standard_en_pos_tag.toDF().show()


# ### Malay Pos Tagging

# In[43]:


language_ms= language_ms.repartition(110)


# In[44]:


language_ms.count()


# In[45]:


sampling_ms = language_ms.limit(10000)


# In[46]:


sampling_ms.count()


# In[47]:


sampling_ms = sampling_ms.repartition(220)


# In[48]:


sampling_ms.count()


# In[49]:


ms_pos_tag = sampling_ms.rdd.map(lambda x: (x[0], x[1], ms_pos_tag(x[0]))).persist(StorageLevel.MEMORY_ONLY)


# In[50]:


ms_pos_tag.toDF().show()


# In[51]:


get_ipython().run_cell_magic('time', '', 'ms_pos_tag.count()\n')


# In[52]:


column = ['sentence', 'language', 'pos_tag']
df_zh = standard_zh_pos_tag.toDF(column)
df_en = standard_en_pos_tag.toDF(column)
df_ms = ms_pos_tag.toDF(column)
df_none = none_pos_tag.toDF(column)


# In[ ]:


df_final = df_zh.union(df_en).union(df_ms).union(df_none).persist(StorageLevel.MEMORY_ONLY)


# In[ ]:


df_final.show()


# In[ ]:


from pyspark.sql.functions import col, concat_ws
df_final = df_final.withColumn('pos_tag', concat_ws(', ', col('pos_tag')))


# In[ ]:


df_final.show()


# In[ ]:


import pyspark.sql.functions as f
df_final = df_final.withColumn('n-gram', f.size(f.split(f.col('pos_tag'), ' ')))


# In[ ]:


df_final = df_final.filter(col('n-gram') < 6)


# In[ ]:


df_final.show()

df_final = df_final.coalesce(1)

df_final.write.parquet('hdfs://g5.bigtop.it:8020/user/root/social_media_final')df_final.write.csv('social_media_final', header = True)
# In[ ]:




