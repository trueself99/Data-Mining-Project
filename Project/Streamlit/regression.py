import streamlit as st
import functions
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import validation_curve
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import SMOTE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import calendar
import imblearn

def app_page():
    st.title("Regression Models")
    st.subheader("Data after One hot encoding and top 5 features selected")
    df_T5 = pd.read_csv('ohe.csv')
    st.write(df_T5.head())
    
    st.subheader("Multinomial Logistic Regression")
    y = df_T5.NUM_OF_BASKETS
    X = df_T5.drop("NUM_OF_BASKETS",1)
    colnames = X.columns
    
    smote = SMOTE(sampling_strategy = 'minority')
    X_sm, y_sm = smote.fit_resample(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.3, random_state=0) #70% train 30% test
    model = OneVsRestClassifier(LogisticRegression(multi_class='multinomial', C=0.15, penalty="l2", solver='lbfgs',max_iter=5000))
    model.fit(X_train, y_train)
    #Predict using model
    y_pred = model.predict(X_test)
    
    #Saving accuracy score in table
    result = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test,y_pred,average='weighted')
    recall = recall_score(y_test,y_pred, average='weighted')
    st.write("Accuracy = ",result)
    st.write("Precision = ", precision)
    st.write("Recall = ", recall)
    
    st.image('images/multiclass.png')
    
    