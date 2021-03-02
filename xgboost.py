# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 12:04:25 2020

@author: Dell
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
import shap
import xgboost as xgb
from xgboost import XGBClassifier


def fonction_select_xgb(data):

   
    X = data.drop("tx_rec_marg_Bin",axis = 1)
    y = data["tx_rec_marg_Bin"]

# préparation de la base on split en 70% training et 30% test #

    X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state = 0)


    kf = KFold(n_splits=3)  
    kf.get_n_splits(X)

   
    rf = SelectFromModel(RandomForestClassifier(n_estimators = 100))
    rf.fit(X_train, y_train)

# on récupére les variables choisis par l'algo dans sa sélection #

    selected_feat = X_train.columns[(rf.get_support())]
   
    sel = list(selected_feat)
   
    target = ["tx_rec_marg_Bin"]
   
    select = sel + target
   

# on trace un petit histo montrant la distribution des variables selectionées #

    #pd.Series(rf.estimator_.feature_importances_.ravel()).hist()
   
   
    return data[select]


def fonction_model_xgb(data):
   
    df = fonction_select_xgb(data)
   
    X = df.drop("tx_rec_marg_Bin",axis = 1)
    y = df["tx_rec_marg_Bin"]
   
    X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state = 0)

    kf = KFold(n_splits=3)  
    kf.get_n_splits(X)



    Quant=df[[col for col in df.columns.to_list() if df[col].nunique() > 3]]
   
    num = list(Quant.columns)
   

    scaler = StandardScaler().fit(X_train[num])
    X_train[num] = scaler.transform(X_train[num])
    X_test[num]  = scaler.transform(X_test[num])
   

    xgb = XGBClassifier(criterion = 'gini',max_depth = 7, max_features = 'auto', n_estimators = 500)
   

    #grid_xgb = GridSearchCV (estimator = xgb, param_grid=param_grid ,scoring="accuracy")

    #print(grid_rf.best_params_)

    xgb.fit(X_train, y_train)


    y_pred = xgb.predict(X_test)

    print(classification_report(y_test,y_pred))
    #print(grid_xgb.best_params_)
   
    xgb_shap = XGBClassifier(criterion = 'gini',max_depth = 7, max_features = 'auto', n_estimators = 500)
   
    xgb_shap.fit(X_train, y_train)
    shap_values = shap.TreeExplainer(xgb_shap).shap_values(X_train)

    print(shap.summary_plot(shap_values, X_train, plot_type="bar"))
    print(confusion_matrix(y_test,y_pred))
    print(f1_score(y_pred,y_test, average='micro'))
    
    return data