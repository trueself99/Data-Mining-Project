# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 03:13:03 2022

@author: Adam Hilman
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
import functions

def app_page():
    st.title("Classification Model")
    top = st.container()
    mainContainer = st.container()
    col1, col2, col3 = st.columns(3)
    bottom = st.container()
    col4,col5=st.columns(2)
    rfeContainer = st.container()
    class_names = ['0', '1']
    
    
    with top:
        option = st.selectbox('What Feature Selection Method would you like to use?',['Random Forest','Naive Bayes'])
    
    if 'Random Forest' in option:
        
        with mainContainer:
        
            with col1:
                #Boruta + SMOTE 
                st.subheader("Random Forest")
                df_T10 = pd.read_csv("df_le.csv")
                df_T10 = df_T10.drop(['GENDER_LE','DAY_TYPE_LE','WASHER_NO_LE','WASH_ITEM_LE','DRYER_NO_LE', 'PANTS_TYPE_LE','BODY_SIZE_LE','RACE_LE','DAY_LE'], axis = 1)

                y = df_T10.BASKET_SIZE_LE
                X = df_T10.drop("BASKET_SIZE_LE",1)
                colnames = X.columns
              
               
                
                #SMOTE data
                smote = SMOTE(sampling_strategy = 'minority')
                X_sm, y_sm = smote.fit_resample(X,y)
    
                    
                #Split data into train and test
                X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.3, random_state=0, stratify=y_sm) #70% train 30% test
                n_estimators=[226]
                bootstrap=[True]
                max_depth=[None]
                
                param_grid={'n_estimators': n_estimators,
                            'bootstrap': bootstrap,
                            'max_depth': max_depth
                            }
                         
                rfModel = RandomForestClassifier()
                rfGrid = GridSearchCV(estimator= rfModel, param_grid=param_grid, cv=3, verbose=2, n_jobs=4)
                rfGrid.fit(X_train, y_train)
                y_predicted = rfGrid.predict(X_test)
                #
                #model evaluation
                #score
                score = rfGrid.score(X_test, y_test)
                
                #AUC
                prob_RF = rfGrid.predict_proba(X_test)
                prob_RF = prob_RF[:, 1]

                auc_RF= roc_auc_score(y_test, prob_RF)
                fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test, prob_RF) 
                     
                plt.plot(fpr_RF, tpr_RF, color='blue', label='RF') 
                plt.plot([0, 1], [0, 1], color='red', linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend()
                     
                #st.set_option('deprecation.showPyplotGlobalUse', False)
                plot_roc_curve(rfGrid, X_test, y_test)
                st.pyplot()
                     
                confusion_majority=confusion_matrix(y_test,y_predicted)
                
                st.write('**Precision=** {:.2f}'.format(precision_score(y_test, y_predicted)))
                st.write('**Recall**= {:.2f}'. format(recall_score(y_test, y_predicted)))
                st.write('**F1=** {:.2f}'. format(f1_score(y_test, y_predicted)))
                st.write('**Accuracy=** {:.2f}'. format(accuracy_score(y_test, y_predicted)))
                st.write('**AUC=** %.2f' % auc_RF)
                     
                #plot_confusion_matrix(rfModel, X_test, y_test, display_labels=class_names)
                #st.pyplot()
                cm = confusion_matrix(y_test,y_predicted)
                st.write("**Confusion Matrix:** ", cm)
        
    if 'Naive Bayes' in option:
        with mainContainer:
                 
            with col1:
                st.subheader("Naive Bayes")
                df_T10 = pd.read_csv("df_le.csv")
                df_T10 = df_T10.drop(['GENDER_LE','DAY_TYPE_LE','WASHER_NO_LE','WASH_ITEM_LE','DRYER_NO_LE', 'PANTS_TYPE_LE','BODY_SIZE_LE','RACE_LE','DAY_LE'], axis = 1)

                y = df_T10.BASKET_SIZE_LE
                X = df_T10.drop("BASKET_SIZE_LE",1)
              
                colnames = X.columns
                
                #SMOTE data
                smote = SMOTE(sampling_strategy = 'minority')
                X_sm, y_sm = smote.fit_resample(X,y)
                
                #Split data into train and test
                X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.3, random_state=0) #70% train 30% test
                nb = GaussianNB(priors = None, var_smoothing = 0.0001)
                nb.fit(X_train, y_train)
                y_pred = nb.predict(X_test)
                
                #Model Evaluation
                score = nb.score(X_test, y_test)
                
                
                prob_NB = nb.predict_proba(X_test)
                prob_NB = prob_NB[:, 1]
                
                auc_NB= roc_auc_score(y_test, prob_NB)
                
                fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, prob_NB) 

                plt.plot(fpr_NB, tpr_NB, color='orange', label='NB') 
                plt.plot([0, 1], [0, 1], color='green', linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend()
                
                #st.set_option('deprecation.showPyplotGlobalUse', False)
                plot_roc_curve(nb, X_test, y_test)
                st.pyplot()
                
                confusion_majority = confusion_matrix(y_test, y_pred)

                st.write('**Precision=** {:.2f}'.format(precision_score(y_test, y_pred)))
                st.write('**Recall**= {:.2f}'. format(recall_score(y_test, y_pred)))
                st.write('**F1=** {:.2f}'. format(f1_score(y_test, y_pred)))
                st.write('**Accuracy=** {:.2f}'. format(accuracy_score(y_test, y_pred)))
                st.write('**AUC=** %.2f' % auc_NB)
                
                #plot_confusion_matrix(nb, X_test, y_test, display_labels=class_names)
                #st.pyplot()
                cm = confusion_matrix(y_test,y_pred)
                st.write("**Confusion Matrix:** ", cm)
           
                    
                