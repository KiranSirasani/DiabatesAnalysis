#!/usr/bin/env python
# coding: utf-8

# Reference: https://towardsdatascience.com/a-beginners-guide-to-data-analysis-in-python-188706df5447
# 
# Sample DataSet
# https://www.kaggle.com/uciml/pima-indians-diabetes-database

# In[1]:


## Data Reading using Pandas and DataFrame
## pip install pandas
## pip install pandas_profiling
## pip install seaborn
## pip install matplotlib
## pip install plotly
## pip install numpy

import pandas as pd
df = pd.read_csv(r'C:\Users\kiran\OneDrive\Desktop\Learning\Data\diabetes.csv')
df.head()


# Columns in Data Frame
# 
# Pregnancies: The number of pregnancies the patient had
# Glucose: The patient’s glucose level
# Blood Pressure
# Skin Thickness: The thickness of the patient’s skin in mm
# Insulin: Insulin level of the patient
# BMI: Body Mass Index of patient
# DiabetesPedigreeFunction: History of diabetes mellitus in relatives
# Age
# Outcome: Whether or not a patient has diabetes
# 

# Types of Variables
# 
# Numeric variables are variables that are a measure, and have some kind of numeric meaning. All the variables in this dataset except for “outcome” are numeric.
# Categorical variables are also called nominal variables, and have two or more categories that can be classified.
# 
# 

# In[2]:


### Data Profiling using pandas

import pandas_profiling as pp
pp.ProfileReport(df)


# In[5]:


# Visualization Imports
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
import numpy as np


# In[6]:


dist = df['Outcome'].value_counts()
colors = ['mediumturquoise', 'darkorange']
trace = go.Pie(values=(np.array(dist)),labels=dist.index)
layout = go.Layout(title='Diabetes Outcome')
data = [trace]
fig = go.Figure(trace,layout)
fig.update_traces(marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.show()


# In[7]:


def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist() }
import plotly.graph_objects as go
dfNew = df.corr()
fig = go.Figure(data=go.Heatmap(df_to_plotly(dfNew)))
fig.show()


# In[8]:


fig = px.scatter(df, x='Glucose', y='Insulin')
fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Glucose and Insulin')
fig.show()


# In[9]:


fig = px.box(df, x='Outcome', y='Age')
fig.update_traces(marker_color="midnightblue",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Age and Outcome')
fig.show()


# In[10]:


plot = sns.boxplot(x='Outcome',y="BMI",data=df)


# In[ ]:




