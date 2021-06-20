#!/usr/bin/env python
# coding: utf-8

#  Name : Chinmayee Taralkar      
#  Roll No : BECOB262

# Problem Statement : Apply a-priori algorithm to find frequently occurring items from given data and generate
# strong association rules using support and confidence thresholds.
# For Example: Market Basket Analysis

# Code :

# #Importing the required libraries

# In[38]:


import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# #Loading the data

# In[6]:


data = pd.read_excel("C:\\Users\\Lenovo\\Downloads\\Online Retail.xlsx")


# In[39]:


data.head()


# #Checking the columns of data

# In[8]:


data.columns


# In[9]:


data.shape


# In[12]:


data.isnull().values.any()


# In[13]:


data.isnull().sum()


# #Cleaning data

# In[40]:


data['Description']=data['Description'].str.strip()
data.dropna(axis=0,subset=['InvoiceNo'],inplace=True)
data['InvoiceNo']=data['InvoiceNo'].astype('str')
data=data[~data['InvoiceNo'].str.contains('C')]


# In[41]:


data.head()


# In[42]:


data['Country'].value_counts()


# In[43]:


data.Country.unique()


# In[64]:


#transactions done in Netherlands
basket_Nether = (data[data['Country'] == "Netherlands"]
                        .groupby(['InvoiceNo','Description'])['Quantity']
                        .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))


# In[20]:


#transactions done in Germany
basket_Germany = (data[data['Country'] == "Germany"]
                        .groupby(['InvoiceNo','Description'])['Quantity']
                        .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))


# In[21]:


#transactions done in France
basket_France = (data[data['Country'] == "France"]
                        .groupby(['InvoiceNo','Description'])['Quantity']
                        .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))


# In[22]:


#transactions done in Spain
basket_Spain = (data[data['Country'] == "Spain"]
                        .groupby(['InvoiceNo','Description'])['Quantity']
                        .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))


# In[65]:


#encoding data
def encode(x):
    if x<=0:
        return 0
    if x>=1:
        return 1
    


# In[66]:


#encoding EIRE
basket_encoded = basket_Nether.applymap(encode)
basket_Nether = basket_encoded


# In[67]:


#encoding Germany
basket_encoded = basket_Germany.applymap(encode)
basket_Germany = basket_encoded


# In[68]:


#encoding France
basket_encoded = basket_France.applymap(encode)
basket_France = basket_encoded


# In[69]:


#encoding Spain
basket_encoded = basket_Spain.applymap(encode)
basket_Spain = basket_encoded


# In[70]:


# Building the model for EIRE
freq_Items_Nether = apriori(basket_Nether, min_support = 0.05, use_colnames = True)  


# In[71]:


# Collecting the inferred rules
rulesNether = association_rules(freq_Items_Nether, metric ="lift", min_threshold = 1) 
rulesNether = rulesNether.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rulesNether.head()


# In[73]:


# Building the model for germany
freq_Items_Germany = apriori(basket_Germany, min_support = 0.05, use_colnames = True)  


# In[74]:


# Collecting the inferred rules
rulesGermany = association_rules(freq_Items_Germany, metric ="lift", min_threshold = 1) 
rulesGermany = rulesGermany.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rulesGermany.head()


# In[75]:


# Building the model for France
freq_Items_France = apriori(basket_France, min_support = 0.05, use_colnames = True)  


# In[78]:


# Collecting the inferred rules
rulesFrance = association_rules(freq_Items_France, metric ="lift", min_threshold = 1) 
rulesFrance = rulesFrance.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rulesFrance.head()


# In[77]:


# Building the model for Spain
freq_Items_Spain = apriori(basket_Spain, min_support = 0.05, use_colnames = True)  


# In[79]:


# Collecting the inferred rules
rulesSpain = association_rules(freq_Items_Spain, metric ="lift", min_threshold = 1) 
rulesSpain = rulesSpain.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rulesSpain.head()


# In[80]:


#extracting rules for EIRE based on condition
rulesNether[(rulesNether['lift']>=2)&(rulesNether['confidence']>=0.3)]


# In[81]:


#extracting rules for Germany based on condition
rulesGermany[(rulesGermany['lift']>=2)&(rulesGermany['confidence']>=0.3)]


# In[62]:


#extracting rules for France based on condition
rulesFrance[(rulesFrance['lift']>=2)&(rulesFrance['confidence']>=0.3)]


# In[63]:


#extracting rules based on condition
rulesSpain[(rulesSpain['lift']>=2)&(rulesSpain['confidence']>=0.3)]

