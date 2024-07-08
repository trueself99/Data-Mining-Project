# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:21:53 2022

@author: 60115
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import streamlit as st


from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler

def app_page():
    st.title("Clustering Model")
    top = st.container()
    mainContainer = st.container()
    col1, col2, col3 = st.columns(3)
    bottom = st.container()
    col4,col5=st.columns(2)
    rfeContainer = st.container()
    class_names = ['0', '1']
    st.title("Clustering")
    st.header("K-means Clustering")
    
    df = pd.read_csv("data/LaundryData_2021_T2.csv")
    
    df1 = df.copy()
    df1.head()
        #change column headers to UPPERCASE\
    df1.columns = df.columns.str.upper()
            
            #Renaming column AGE_RANGE to AGE
    df1 = df1.rename(columns = {"AGE_RANGE": "AGE"})
            
            #Replace AGE null with mean age
    age_Mean = df1["AGE"].mean()
    df1["AGE"] = df1["AGE"].fillna(int(age_Mean))
    df1["NUM_OF_BASKETS"] = df1["NUM_OF_BASKETS"].fillna(int(age_Mean))
            
            #Replace categorical null with mode (race, body size, kids, basket size etc)
    df1['RACE'] = df1['RACE'].fillna(df1['RACE'].value_counts().index[0])
    df1['GENDER'] = df1['GENDER'].fillna(df1['GENDER'].value_counts().index[0])
    df1['BODY_SIZE'] = df1['BODY_SIZE'].fillna(df1['BODY_SIZE'].value_counts().index[0])
    df1['WITH_KIDS'] = df1['WITH_KIDS'].fillna(df1['WITH_KIDS'].value_counts().index[0])
    df1['KIDS_CATEGORY'] = df1['KIDS_CATEGORY'].fillna(df1['KIDS_CATEGORY'].value_counts().index[0])
    df1['BASKET_SIZE'] = df1['BASKET_SIZE'].fillna(df1['BASKET_SIZE'].value_counts().index[0])
    df1['BASKET_COLOUR'] = df1['BASKET_COLOUR'].fillna(df1['BASKET_COLOUR'].value_counts().index[0])
    df1['ATTIRE'] = df1['ATTIRE'].fillna(df1['ATTIRE'].value_counts().index[0])
    df1['SHIRT_COLOUR'] = df1['SHIRT_COLOUR'].fillna(df1['SHIRT_COLOUR'].value_counts().index[0])
    df1['SHIRT_TYPE'] = df1['SHIRT_TYPE'].fillna(df1['SHIRT_TYPE'].value_counts().index[0])
    df1['PANTS_COLOUR'] = df1['PANTS_COLOUR'].fillna(df1['PANTS_COLOUR'].value_counts().index[0])
    df1['PANTS_TYPE'] = df1['PANTS_TYPE'].fillna(df1['PANTS_TYPE'].value_counts().index[0])
    df1['WASH_ITEM'] = df1['WASH_ITEM'].fillna(df1['WASH_ITEM'].value_counts().index[0])
    df1['SPECTACLES'] = df1['SPECTACLES'].fillna(df1['SPECTACLES'].value_counts().index[0])
            
    #Removing '_' from column values
    df1["KIDS_CATEGORY"] = df1["KIDS_CATEGORY"].str.replace("_", " ")
    df1["SHIRT_TYPE"] = df1["SHIRT_TYPE"].str.replace("_", " ")
            
    #Change blue_jeans to blue
    df1["PANTS_COLOUR"] = df1["PANTS_COLOUR"].replace(["blue_jeans"], "blue")
            
    #clean time column ';'
    df1["TIME"] = df1["TIME"].str.replace(";", ":")
            
            #Add "DATE_TIME" column
    df1["DATE_TIME"] = df1["DATE"] + " " + df1["TIME"]
    df1["DATE_TIME"] = pd.to_datetime(df1['DATE_TIME'])
            
            
    #Add "DAY" and "WEEKDAY" column
    day = df1["DATE_TIME"].dt.dayofweek
    df1["DAY"] = df1["DATE_TIME"].dt.day_name()
    df1["DAY_TYPE"] = (day < 5).astype(str)
    df1["DAY_TYPE"] = df1["DAY_TYPE"].str.replace("True", "weekday")
    df1["DAY_TYPE"] = df1["DAY_TYPE"].str.replace("False", "weekend")
            
    x = df1[['LATITUDE','LONGITUDE']]
    ss = StandardScaler()
    X = ss.fit_transform(x)
           
    print(X)
            
    km = KMeans(n_clusters = 5, random_state=1)
    km.fit(X)
    y_km = km.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_km, s=50, cmap='viridis')

    centers = km.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    st.pyplot()
   