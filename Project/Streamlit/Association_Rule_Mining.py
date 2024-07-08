# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 04:45:51 2022

@author: 60115
"""
import streamlit as st
import pandas as pd
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth

def app_page():
    top = st.container()
    mainContainer = st.container()
    col1, col2, col3 = st.columns(3)
    bottom = st.container()
    col4,col5=st.columns(2)
    rfeContainer = st.container()
    class_names = ['0', '1']

    st.title("Association Rule Mining")
            
    df = pd.read_csv("data/LaundryData_2021_T2.csv")
        
        #Convert data to boolean features
    
    te = TransactionEncoder()
    te_array = te.fit(df).transform(df)
    df1 = pd.DataFrame(te_array, columns=te.columns_)
    
    #Find frequently occurring itemsets using Apriori Algorithm
    frequent_itemsets_ap = apriori(df1, min_support=0.001, use_colnames=True)
    st.write(' test'.format(frequent_itemsets_ap))
    
    #Find frequently occurring itemsets using F-P Growth
    frequent_itemsets_fp=fpgrowth(df1, min_support=0.001, use_colnames=True)
    st.write(' '.format(frequent_itemsets_fp))
    
    #Mine the Association Rules
    rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=0.8)
    rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.8)
    st.write('AP rules= '.format(rules_ap))
    st.write('FP rules= '.format(rules_fp))
    
