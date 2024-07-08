import streamlit as st
import pandas as pd
import functions

def app_page():

    # Print title
    st.title("TDS 3301- Data Mining 🧪")
    st.subheader("The Data")
    st.header('Here is the dataset used in the Assignment')
    
    df = functions.getDataset()
    st.write(df.head())

    st.markdown(
        f'''
        This application serves as a application and find it useful for your understanding of the simple, yet powerful tool that is the Naive-Bayes classifier.
        '''
    )
        