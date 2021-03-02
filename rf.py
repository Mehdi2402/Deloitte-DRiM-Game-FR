#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 12:03:10 2020

@author: ahmadcharaf
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import shap




def fonction_select(data):

    
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
    
    
    return data[select]


def fonction_model(data):
    
    df = fonction_select(data)
    
    X = df.drop("tx_rec_marg_Bin",axis = 1)
    y = df["tx_rec_marg_Bin"] 
    
    X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state = 0)

    kf = KFold(n_splits=3)   
    kf.get_n_splits(X)



    QuantVar=df[[col for col in df.columns.to_list() if df[col].nunique() > 3]]
    num = list(QuantVar.columns)

    scaler = StandardScaler().fit(X_train[num])
    X_train[num] = scaler.transform(X_train[num])
    X_test[num]  = scaler.transform(X_test[num])

    rf = RandomForestClassifier()

    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'log2'],
        'max_depth' : [5,6,7],
        'criterion' :['gini', 'entropy']
    }
    

    grid_rf = GridSearchCV (estimator = rf, param_grid=param_grid ,scoring="accuracy")

    grid_rf.fit(X_train, y_train)


    y_pred = rf.predict(X_test)
    
    rf_shap = RandomForestClassifier(criterion=grid_rf.best_params_["criterion"]
                            , max_depth=grid_rf.best_params_["max_depth"]
                            , max_features=grid_rf.best_params_["max_features"]
                            ,n_estimators=grid_rf.best_params_["n_estimators"])
    
    print(grid_rf.best_params_)
    print(classification_report(y_test,y_pred))
    
    rf_shap.fit(X_train, y_train)
    shap_values = shap.TreeExplainer(rf_shap).shap_values(X_train)
    
    print(confusion_matrix(y_test,y_pred))
    print(shap.summary_plot(shap_values, X_train, plot_type="bar"))
    
    return data