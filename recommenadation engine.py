# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:56:05 2023

@author: lenovo
"""



**Problem statement:-

Build a recommender system by using cosine simillarties score.



#Importing the Necessary Liabrary
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt


#Loading the Dataset
df=pd.read_csv('C:/Users/lenovo/OneDrive/Documents/EXCLER CODES AND DATSETS/Recommendation Engine/Movie.csv')

#EDA
df.head()
df.tail()
df.shape
df.info()
df.describe()
df.isnull().sum()
#number of unique users in the dataset
len(df.userId.unique())
len(df.movie.unique())

#Prepare the Pivot table.
user_movies_df = df.pivot(index='userId',
                                 columns='movie',
                                 values='rating').reset_index(drop=True)

#Impute those NaNs with 0 values
user_movies_df.fillna(0, inplace=True)
user_movies_df

#Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


user_sim = 1 - pairwise_distances( user_movies_df.values,metric='cosine')
user_sim


#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)


#Set the index and column names to user ids 
user_sim_df.index = df.userId.unique()
user_sim_df.columns = df.userId.unique()

user_sim_df.iloc[0:5, 0:5]

np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]

#Most Similar Users
user_sim_df.idxmax(axis=1)[0:5]

df[(df['userId']==6) | (df['userId']==168)]


user_1=df[df['userId']==6]
user_2=df[df['userId']==11]
user_2


user_1.movie

#Merge 
pd.merge(user_1,user_2,on='movie',how='outer')
