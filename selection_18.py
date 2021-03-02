# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:31:26 2020

@author: nelso
"""

from DRIM.libraries import *
from sklearn import preprocessing

chemin_DRIM = r'C:\Users\nelso\OneDrive\Bureau\del'
os.chdir(r'{}\Codes'.format(chemin_DRIM))

# Importer la base
mat18 = pd.read_csv(r'C:\Users\nelso\OneDrive\Bureau\mat18.csv', sep=';' )

#Drop des variables non explicatives
mat18 = mat18.drop(['cle2', 'dte_def_2', 'dte_sor_2', 'dtf_per_trt', '_maturity_', '_clos_',
       'rcvt_act_c', 'tx_rec_act_2','cat_seg_labels', 'CD_CAT_EXPO_4_labels', 'qual_veh_labels',
       'nat_veh_labels'], axis=1)

#####Sinon problème avec le calcul du ratio_ead
mat18=mat18[mat18["MT_INI_FIN_"]!=0]

####Matrice corrélation
QuantVar=mat18[[col for col in mat18.columns.to_list() if mat18[col].nunique() > 6]]

corrcont(QuantVar)

###############To drop après étude des corrélations
mat18=mat18.drop(columns=["ead","dur_b_defm_","mt_appo_","dur_b_endm_",
                          "mt_appo_",'MT_INI_FIN__cat_seg','mt_appo__qual_veh','chom_euro_area',
                          'DUR_PREV_FIN_qual_veh','tic_euro_area',"cpi_euro_area","cli_euro_area",
                          "cpi_spain", "tic_spain", "MT_INI_FIN__qual_veh",\
                              "DUR_PREV_FIN_cat_seg",'DUR_PREV_FIN_CD_CAT_EXPO_4',
                              "cat_seg","ratio_ead"],axis=1)
mat18=mat18[mat18["outliers"]==0]
mat18=mat18.drop(["tweaked_maturity","outliers"], axis=1)
################	


'''
bin_values = np.arange(start=0.01, stop=1.01, step=0.05)
mat18["tx_rec_marg"].hist(bins=bin_values, figsize=[14,6])

mat18["tx_rec_marg"].quantile([0,0.25,0.5,0.75,1])
'''

#Discrétisation de la target
median_bin = list(mat18[mat18["tx_rec_marg"]>0.05]["tx_rec_marg"].describe())[5]
# c'est la medianne des nombre hors 0
mat18['tx_rec_marg_Bin'] = pd.cut(mat18['tx_rec_marg'] , [-1,0.05,median_bin,1] , labels = [0,1,2])
mat18['tx_rec_marg_Bin'] =  mat18['tx_rec_marg_Bin'].astype(float)


#Standardisation des variables numériques
Standardize=['DUR_PREV_FIN', 'MT_INI_FIN_', 'mt_appo__cat_seg', 'mt_appo__CD_CAT_EXPO_4',\
             'DUR_PREV_FIN_CD_CAT_EXPO_4',"ead_cat_seg",'ead_qual_veh',"chom_spain","cli_spain"]
    

ToStandardize=mat18.drop([c for c in mat18.columns if c not in Standardize], axis=1)
mat18=mat18.drop([c for c in mat18.columns if c in Standardize], axis=1)

scaler = preprocessing.MinMaxScaler().fit(ToStandardize)
ToStandardize_scaled = scaler.transform(ToStandardize)
ToStandardize_scaled=pd.DataFrame(ToStandardize_scaled, columns=ToStandardize.columns)

Tokeep=list(list(mat18.columns) + list(ToStandardize.columns))
mat18=pd.concat([mat18.reset_index(drop=True),ToStandardize_scaled.reset_index(drop=True)],axis=1,ignore_index=True)
mat18.columns=Tokeep

X = mat18.drop(["tx_rec_marg","tx_rec_marg_Bin"], axis=1)
y = mat18['tx_rec_marg_Bin']

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size= 0.33 , random_state=42)

#RFECV stratifié pour garder intact la proportion de target dans l'échantillon d'entraînement
def RFECVRF(X_train,y_train,NumberOfStep,Kfold):
    '''

    Parameters
    ----------
    X_train : Train dataset features.
    y_train : Train dataset target.
    NumberOfStep : Number of features removed by each step of the algorithm.
    Kfold : Number of split in our dataset in order to do cross validation.

    Returns
    -------
    Model fiting, by nothing really returns.

    '''
    
    
    step=NumberOfStep
    cv=StratifiedKFold(Kfold)
    clf_rf=LogisticRegression(multi_class='multinomial', class_weight='balanced',solver='lbfgs',max_iter=300)
    rfecv=RFECV(estimator=clf_rf, step=step, cv=cv, scoring='roc_auc_ovr_weighted')
    
    return rfecv


def RFECVplot(X_train,y_train):
    
    '''
    Input : We use the RFECVRF function
    
    Output: It return the plot of RFECV in order to know best number of features, it returns
    the Optimal number of features and the RFECV model itself
    
    '''
        
    rfecv=RFECVRF(X_train,y_train,1,5)
    rfecv.fit(X_train,y_train)

    print("Number of optimal features :",  rfecv.n_features_)
    OptiNbFeatures=rfecv.n_features_
    
    plt.figure(figsize=(20,10))
    plt.xlabel('Number of features selected')
    plt.ylabel('CV score')
    plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
    
    return rfecv, OptiNbFeatures
  

#Application à notre modèle choisi 

def RFERF(X,X_train, y_train,X_test,NumberOfStep,OptiNbFeatures):
    '''
    Inputs
    ----------
    X_train : .
    y_train : .
    X_test : .
    NumberOfStep : TNumber of features removed by each step of the algorithm.
    OptiNbFeatures : We got it from the RFECVplot function.

    Returns
    -------
    X_train : Train dataset with selected features.
    X_test : Test dataset with selected features.
    FeatList : List of features we kept.

    '''
    
    clf_rf=LogisticRegression(multi_class='multinomial', class_weight='balanced' ,solver='lbfgs',max_iter=300)
    rfe_sel = RFE(estimator=clf_rf, step=NumberOfStep, n_features_to_select=OptiNbFeatures)
    rfe_sel_fit = rfe_sel.fit(X_train, y_train)
    print(rfe_sel_fit.support_)
    print(rfe_sel_fit.ranking_)

    KeptFeatures = list(X.columns)     
    FeaturesFilt = pd.Series(rfe_sel_fit.support_,index = KeptFeatures)
    selected_features_rfe = FeaturesFilt[FeaturesFilt==True].index
    print(type(selected_features_rfe))
    
    FeatList=list(selected_features_rfe)
    
    print("Final Features :", FeatList)

    X_train= X_train[FeatList]
    X_test= X_test[FeatList]
    
    return X_train, X_test, FeatList

#Sélection de variables
rfecv, OptiNbFeatures=RFECVplot(X_train,y_train)
X_train, X_test, FeatList= RFERF(X,X_train, y_train,X_test,1,11)

import sklearn
from sklearn.metrics import *
#Test rapide des résultats avant modélisation sur SAS
model=LogisticRegression(fit_intercept=True, class_weight='balanced' ,solver='lbfgs',max_iter=300)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
y_proba= model.predict_proba(X_test)
y_proba = y_proba[:, 1]

cm=pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])

print('Accuracy of logistic regression classifier on test set: {:.4f}'.\
      format((cm[0][0]+cm[1][1]+cm[2][2])/cm.to_numpy().sum()))
    

#Exportation pour modélisation SAS
modelfeat= list(X_train.columns)
modelfeat.append("tx_rec_marg_Bin")
modelfeat.append("tx_rec_marg")

mat18=mat18[modelfeat]

mat18.to_csv(r"C:\Users\nelso\OneDrive\Bureau\mat18nodiscr.csv")