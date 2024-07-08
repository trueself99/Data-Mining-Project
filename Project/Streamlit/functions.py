import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import functions

def getDataset():
    df = pd.read_csv('data/LaundryData_2021_T2.csv')
    return df

def cleanData(df):
    df1 = df.copy()
    #change column headers to UPPERCASE
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
    return df1

def labelEncoder(df1):
    df_laundry = df1.copy()
    df_le = df_laundry.copy()
    
    df_le['RACE_LE'] = LabelEncoder().fit_transform(df_laundry.RACE)
    df_le['GENDER_LE'] = LabelEncoder().fit_transform(df_laundry.GENDER)
    df_le['BODY_SIZE_LE'] = LabelEncoder().fit_transform(df_laundry.BODY_SIZE)
    df_le['WITH_KIDS_LE'] = LabelEncoder().fit_transform(df_laundry.WITH_KIDS)
    df_le['KIDS_CATEGORY_LE'] = LabelEncoder().fit_transform(df_laundry.KIDS_CATEGORY)
    df_le['BASKET_SIZE_LE'] = LabelEncoder().fit_transform(df_laundry.BASKET_SIZE)
    df_le['BASKET_COLOUR_LE'] = LabelEncoder().fit_transform(df_laundry.BASKET_COLOUR)
    df_le['ATTIRE_LE'] = LabelEncoder().fit_transform(df_laundry.ATTIRE)
    df_le['SHIRT_COLOUR_LE'] = LabelEncoder().fit_transform(df_laundry.SHIRT_COLOUR)
    df_le['SHIRT_TYPE_LE'] = LabelEncoder().fit_transform(df_laundry.SHIRT_TYPE)
    df_le['PANTS_COLOUR_LE'] = LabelEncoder().fit_transform(df_laundry.PANTS_COLOUR)
    df_le['PANTS_TYPE_LE'] = LabelEncoder().fit_transform(df_laundry.PANTS_TYPE)
    df_le['WASH_ITEM_LE'] = LabelEncoder().fit_transform(df_laundry.WASH_ITEM)
    df_le['WASHER_NO_LE'] = LabelEncoder().fit_transform(df_laundry.WASHER_NO)
    df_le['DRYER_NO_LE'] = LabelEncoder().fit_transform(df_laundry.DRYER_NO)
    df_le['SPECTACLES_LE'] = LabelEncoder().fit_transform(df_laundry.SPECTACLES)
    df_le['DAY_LE'] = LabelEncoder().fit_transform(df_laundry.DAY)
    df_le['DAY_TYPE_LE'] = LabelEncoder().fit_transform(df_laundry.DAY_TYPE)
    
    df_le = df_le.drop(['RACE', 'GENDER', 'BODY_SIZE', 'WITH_KIDS', 'KIDS_CATEGORY', 'BASKET_SIZE', 'BASKET_COLOUR', 'ATTIRE'
                        , 'SHIRT_COLOUR', 'SHIRT_TYPE', 'PANTS_COLOUR', 'PANTS_TYPE', 'WASH_ITEM', 'WASHER_NO'
                        , 'DRYER_NO', 'SPECTACLES', 'DAY', 'DAY_TYPE','DATE','TIME','DATE_TIME', 'LONGITUDE', 'LATITUDE'], axis=1)
    return df_le

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

def boruta(X, y, colnames):
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth = 5)

    feat_selector = BorutaPy(rf, n_estimators="auto", random_state = 1)

    st.write("Selecting Features...")
    feat_selector.fit(X.values, y.values.ravel())
    st.write("Ranking Features...")
    boruta_score = functions.ranking(list(map(float, feat_selector.ranking_)), colnames, order=-1)
    boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
    boruta_score = boruta_score.sort_values("Score", ascending = False)
    st.write("Done!")
    
    return boruta_score

def rfe(X,y,colnames):

    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth = 5, n_estimators = 100)
    rf.fit(X,y)        
    rfe = RFECV(rf, min_features_to_select = 1, cv = 3)
    st.write("Selecting Features...")
    rfe.fit(X, y)
    st.write("Ranking Features...")
    rfe_score = functions.ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
    rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
    rfe_score = rfe_score.sort_values("Score", ascending = False)
    return rfe_score































