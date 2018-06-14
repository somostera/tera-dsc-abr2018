
# coding: utf-8

# In[6]:


import pandas as pd
pd.set_option('max_columns', 500)


# In[7]:


df = pd.read_excel('AmesHousing.xls')


# In[8]:


df.head()


# Quer ajuda? Veja a explicação dos dados (e o que são outliers de verdade e podem ser removidos) [aqui](https://ww2.amstat.org/publications/jse/v19n3/decock/DataDocumentation.txt)
