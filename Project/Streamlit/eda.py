import streamlit as st
import pandas as pd
import numpy as np
import functions
import matplotlib.pyplot as plt
import seaborn as sns

def app_page():
    
    data = st.container()
    
    with data:
        st.title("TDS 3301- Data Mining 🧪")
        st.subheader("The Data")
        st.subheader('Here is the dataset used in the Assignment')
        df = functions.getDataset()
        st.write(df.head())
        st.write(" ")
        #st.write("The table above represents the first few rows of the **decoded** dataset. Other than decoding, we have filled the empty cells with *NULL*. We also renamed **ANOTHER CASE** to **ANOTHER_CASE** to follow the format of the other column names.")
    
        df1 = functions.cleanData(df)
        st.subheader("Cleaned Dataset")
        st.write(df1.head())
        st.write("Above is the cleaned dataset, we have added a few columns such as Daytype, day and date time. A few attributes have also been ammended.")

        st.header("Exploratory Data Analysis")
        st.subheader("Age Chart")
        analysis = df1.copy()

        df_ages = analysis['AGE']
        
        analysis['AGE_RANGE'] = pd.cut(x=df_ages, bins=[10,20,30,40,50,60])
        analysis['AGE_RANGE'].value_counts()
        
            
        
        fig1, ax = plt.subplots(figsize=(15,5))
        age = sns.countplot(x = "AGE_RANGE", data = analysis, color = '#0E1E3D') 
        age.set_xlabel("Age Range")
        #sns.set(rc = {'figure.figsize':(10,10)})
        st.pyplot(fig1)
        st.write("The chart above shows the distribution of customers according to their age. We can observe in this chart that the data is **slightly skewed to the left.**")
        
        st.subheader("Day Chart")
        
        fig_dims = (15,5)
        fig2, ax = plt.subplots(figsize = fig_dims)
        intubated = sns.countplot(data = analysis
                                 , x = 'DAY'
                                 , ax = ax
                                 , color = '#0E1E3D')
        intubated.set_xlabel("Day")
        st.pyplot(fig2)
        
        
        st.subheader("Gender")
        
        fig_dims = (15,5)
        fig2, ax = plt.subplots(figsize = fig_dims)
        intubated = sns.countplot(data = analysis
                                 , x = 'GENDER'
                                 , ax = ax
                                 , color = '#0E1E3D')
        intubated.set_xlabel("Gender")
        st.pyplot(fig2)
        
        st.subheader("Race Chart")
        
        fig_dims = (15,5)
        fig2, ax = plt.subplots(figsize = fig_dims)
        intubated = sns.countplot(data = analysis
                                 , x = 'RACE'
                                 , ax = ax
                                 , color = '#0E1E3D')
        intubated.set_xlabel("Race")
        st.pyplot(fig2)
        
        st.header('Feature Selection Methods')
        st.subheader('Boruta')
        st.write("BORUTA feature selection metho was used in the classification model.")
        
            
        boruta_score = pd.read_csv("boruta.csv")
        rfe_score = pd.read_csv("rfe.csv")
       
        st.write("**BORUTA Top 10 Features**")
        fig3, ax = plt.subplots()
        boruta = sns.catplot(x="Score", y="Features", data = boruta_score[0:10], kind = "bar", palette='crest',
                             height = 14, aspect=2)
        
        st.pyplot(boruta)
        
        
        
        st.write("**RFE Top 10 Features**")
        fig4, ax = plt.subplots()
        RFE = sns.catplot(x="Score", y="Features", data = rfe_score[0:10], kind = "bar", palette='crest',
                             height = 14, aspect=2)
        
        st.pyplot(RFE)
   
    
    
    