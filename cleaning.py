# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:48:28 2020

@author: mehdi
"""
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
from datetime import datetime
from functools import reduce

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import *
from typing import Union
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg

def add_fusion_import(df,fusion):
    p = fusion[fusion['cle2'].isin(df['cle2'].tolist())]
    df = df.merge(p,how='inner',right_on='cle2',left_on='cle2',validate='1:1')
    return df


def valeur_ref(df):
    seuils  = {"cli_spain" : 91.2 , "chom_spain" : 9.1 , "tic_spain" : 4.7 , "cpi_spain" : 90.5 , "cli_euro_area" : 91.2 ,
           "chom_euro_area" : 7.2,"tic_euro_area":  3.9,"cpi_euro_area" : 91.5}
    for k in seuils.keys():
        df[k] = df[k]/seuils[k]
    
    return df


def anomalie(df,num_contrat):
    
    z = df[df['cle2']==num_contrat]
    z = z[['tx_rec_act_2','cle2']]
    
    df_group = pd.DataFrame(z.groupby(["cle2"])['tx_rec_act_2'].diff())

    df_g = df_group[df_group["tx_rec_act_2"] < 0]
    
    return True if len(df_g) else False

def add_fusion(df):
    to_select = []
    unique_c = df['cle2'].unique().tolist()
    for c in unique_c :
        if anomalie(df,c) :
            to_select.append(c)

    df['fusion'] = np.where(df['cle2'].isin(to_select),1,0)

    return df


def add_crise(df,chemin_DRIM):
    crise = pd.read_excel(r'{}\Base\crise.xlsx'.format(chemin_DRIM))
    crise['date'] = parse_date_inter(crise,'date')
    crise = select_timeframe_inter(crise)
    
    df = df.merge(crise,how='left',right_on='date',left_on='dte_def_2',validate='m:1')
    df.drop(columns=['date'],inplace=True)
    return df


def throw_very_censored(df):
    df2=df.copy()
    df2['dte_def_2'] =  pd.to_datetime(df2['dte_def_2'], format='%d/%m/%Y')
    ContThrow=df2[(df2["dte_def_2"] > datetime(2015, 10, 31)) & (df["_clos_"]==0)]
    ToThrow=list(ContThrow["cle2"].unique())
    
    Tokeep = [x for x in df2['cle2'].unique().tolist() if x not in ToThrow]
    
    df = df[df['cle2'].isin(Tokeep)]
    
    return df


    
def parse_date_inter(df,col):
    """
    Cette fonction permet d'appliquer le format date pour les colonnes dates formattées en str

    """
    zz = df[col].apply(lambda x :datetime.strptime(x,'%Y-%m'))
    return pd.to_datetime(zz, format='%Y-%m') + MonthEnd(1)





def select_timeframe_inter(df):
    """
    Cette fonction permet de limiter l'intervalle de temps à la période qui nous intéresse
    On l'appliquera aux tables macros
    """
    
    return df[(df['date']>=datetime.strptime('2009-01-31','%Y-%m-%d')) & (df['date']<=datetime.strptime('2017-04-30','%Y-%m-%d'))]






def import_celan_macro_tables(chemin_DRIM):
    
    """
    Cette fonction permet d'importer puis nettoyer les tables macros
    Le cleaning appliquer est le suivant :
                                        uniformisation des noms date
                                        formattage du type de colonnes
                                        limitation de l'intervalle de temps
                                        selection des colonnes qui nous interesse
                                        Separation spain /euro_area
    """
    
    cli = pd.read_csv(r'{}\Base\CLI.csv'.format(chemin_DRIM),sep=';')
    chom =  pd.read_csv(r'{}\Base\Unemployment-rate.csv'.format(chemin_DRIM),sep=';')
    cpi = pd.read_excel(r'{}\Base\consumer_price_indices.xlsx'.format(chemin_DRIM))
    
    tics = pd.read_csv(r'{}\Base\TI_conso_spain.csv'.format(chemin_DRIM))
    tice = pd.read_csv(r'{}\Base\TI_conso_euro_area.csv'.format(chemin_DRIM))
    
    dict_month = {'Dec' : '12', 'Nov':'11','Oct':'10' , 'Sep':'09' , 'Aug':'08',
              'Jul' : '07', 'Jun':'06' , 'May':'05' , 'Apr' : '04' , 'Mar':'03',
              'Feb' : '02' , 'Jan': '01'}
    
    tics['date'] = tics['date'].apply(lambda x : x[:4] + '-' + dict_month[x[4:]] )
    tice['date'] = tice['date'].apply(lambda x : x[:4] + '-' + dict_month[x[4:]] )

    
    cli.rename(columns={'TIME':'date'},inplace=True)
    chom.rename(columns={'TIME':'date'},inplace=True)
    

    cli['date'] = parse_date_inter(cli,'date')
    tics['date'] = parse_date_inter(tics,'date')
    tice['date'] = parse_date_inter(tice,'date')
    chom['date'] = parse_date_inter(chom,'date')
    cpi['date'] = parse_date_inter(cpi,'date')
    
    chom = select_timeframe_inter(chom)
    cli = select_timeframe_inter(cli)
    tics = select_timeframe_inter(tics)
    tice = select_timeframe_inter(tice)
    cpi = select_timeframe_inter(cpi)
    
    chom_spain = chom[chom['LOCATION']=='ESP'][['date','Value']].rename(columns={'Value':'chom_spain'})
    chom_euro_area = chom[chom['LOCATION']=='EA19'][['date','Value']].rename(columns={'Value':'chom_euro_area'})
    
    cli_spain = cli[cli['LOCATION']=='ESP'][['date','Value']].rename(columns={'Value':'cli_spain'})
    cli_euro_area = cli[cli['LOCATION']=='EA19'][['date','Value']].rename(columns={'Value':'cli_euro_area'})
    
    
    cpi_spain = cpi.drop(columns=['cpi_euro_area'])
    cpi_euro_area = cpi.drop(columns=['cpi_spain'])
    
    return  chom_spain, chom_euro_area,\
            cli_spain, cli_euro_area,\
            tics, tice,\
            cpi_spain , cpi_euro_area
    
    
   
    
def add_macro_features(df,dfs):
    """
    Cette fonction permet de joindre les dfs macros avec notre df en se basant sur la date :dtf_per_trt

    """
    to_join = reduce(lambda left,right: pd.merge(left,right,on='date'), dfs)
    
    df['dte_def_2'] = df['dte_def_2'].apply(lambda x :datetime.strptime(x,'%d/%m/%Y'))    
    df = df.merge(to_join,how='left',right_on='date',left_on='dte_def_2',validate='m:1')
    df.drop(columns=['date'],inplace=True)
    
    return df



def add_stat_features(df):
    """
    Cette fonction permet de rajouter le features suivantes :
        
        no_appo : variable binaire = (1 pour 0 en apport /  0 pour un apport non nul)
        
        ratio_ead : proportion = ead/MT_INI_FIN_
                    (à voir pour cette vaiable car on va avoir une forte correlation positive avec ead
                                                     et une forte correlation négative avec MT_INI_FIN_)
                    
        date_neg : variable binaire = (1 pour ligne qui contient une valeur négative pour dur_b_endm_ / 0 sinon)
                    Cette variable permettra de localiser les lignes avec des valeurs négatives dans la variable
                    dur_b_endm_ pour essayer de contrecarrer le biais que cela créera dans le modèle.
                    
        
    --------------------- A FAIRE ----------------------
    travailler sur la variable ratio_b_endm
    """
    
    
    df['no_appo'] = np.where(df['mt_appo_']==0 , 1 , 0)
    df['ratio_ead'] = df['ead'] / df['MT_INI_FIN_']
    df['date_neg'] = np.where(df['dur_b_endm_']<0 , 1 , 0)

    return df


def label_encode(df,col):
    """
    Cette fonction permet de label encoder la variable col dans le dataframe df
    Elle classe les labels selon leur moyenne du tx_rec_marg

    """
    df[col+'_labels'] = df[col].tolist()
    sorted_labels = df.groupby(col)['tx_rec_marg'].mean().sort_values().index.tolist()
    mapper = dict(zip(sorted_labels,list(range(len(sorted_labels)))))
    for k in mapper.keys():
        df[col] = np.where(df[col]==k , mapper[k] , df[col] )
        
    return df


# def label_encode_robust(df,col):
#     labels = df[col].value_counts().index.tolist()
#     samples = [ df[df[col]==l]['tx_rec_marg'] for l in labels ]
#     p_values = { (a,b) : stats.ttest_ind(samples[a],samples[a+1:][b])[1] \
#                  for a in range(len(samples)-1) for b in range(len(samples[a+1:])) }
#     to_assemble = [ p for p in p_values.keys() if p_values[p]>0.1 ]
     
#     if not to_assemble:
#         return label_encode(df,col)
#     else:
#         for m in list(set([ x for y in to_assemble for x in y])):
#             in_m =[m]
#             for l in to_assemble:
#                 if m in l:
#                     l.remove(m)
#                     in_m.append(l)
#             labels_assemble = [labels[x] for x in in_m]
             
     
def croiser (df , cat , cont):
    df[cont+'_'+cat] = df[cat] * df[cont]
    return df


def croiser_ou_non(df , cat , cont):
    labels = df[cat].value_counts().index.tolist()
    samples = [ df[df[cat]==l][cont] for l in labels ]
    p_value = stats.ttest_ind(samples[0],samples[1])[1]

    return True if p_value <=0.05 else False

def croiseur(df, list_cat , list_cont):
    for cont in list_cont:
        for cat in list_cat:
            if croiser_ou_non(df , cat , cont):
                df = croiser(df , cat , cont)
    return df

#list_cat = ['cat_seg','CD_CAT_EXPO_4','qual_veh']
#list_cont = ['MT_INI_FIN_','mt_appo_','DUR_PREV_FIN']
#mat6 = croiseur(mat6, list_cat , list_cont)

    
def locate_outliers_dbscan(data, columns = ['ead','MT_INI_FIN_','mt_appo_']):
    scaler = MinMaxScaler() 
    df = scaler.fit_transform(data[columns])
    
    outlier_detection = DBSCAN(eps = 0.1, metric="euclidean", 
                                 min_samples = 5,
                                 n_jobs = -1)
    clusters = outlier_detection.fit_predict(df)
    
    return np.where(clusters==-1)[0].tolist()
    

def locate_outliers_zscore(data , columns = ['ead','MT_INI_FIN_','mt_appo_'] , treshold = 3):
    """
    Cette fonction permet de localiser les valeurs abberantes, 
    Elle prend argument un dataframe <data>, une liste de colonne <columns> et 
    un seuil du zscore <treshold>.
    Elle renvoie un dictionnaire, qui contient pour chaque colonne, les indices où
    se trouvent les valeurs aberrantes.
    
    Paramètres :
            Dataframe : pd.DataFrame 
            liste de colonnes : list[str]
            seuil : float
            
    returns :
            Dictionnaire : dict{ str : list[int] }

    """
    df = data[columns]
    z_score = np.abs(stats.zscore(df))
    outliers= np.where(z_score>treshold , True , False)
    
    r , c = np.where(outliers)
    c = [columns[x] for x in c]
    c2 = list(set(c))
    
    mapper = { i : [r[j] for j in range(len(c)) if i == c[j]] for i in c2  }
                
    
    return mapper


def show_outliers_zscore(df,mapper):
    """
    Cette fonction permet de montrer les valeurs aberrantes par colonne.
    Elle prend comme paramètres le df et le dictionnaire des index.
    Elle renvoi un dictionnaire qui contient pour chaque colonne les valeurs aberrantes.
    """
    return { k : df.loc[mapper[k],k].tolist() for k in mapper }



def delete_outliers(df , mapper : Union[dict , list] ):
    """
    Cette fonction permet de supprimer les lignes qui contiennent des valeurs manquantes
    
    """
    if type(mapper) == dict :
        to_drop = list(set([x for k in mapper for x in mapper[k]]))
    else:
        to_drop = mapper
        
    df.reset_index(drop = True , inplace= True)
    
    return df.drop(index = to_drop)

def treat_outliers(df ,mapper : Union[dict , list] ):
    """
    Cette fonction permet de créer une variable indicatrice <outliers> qui localise les outliers
    
    """
    if type(mapper) == dict :
        to_treat = list(set([x for k in mapper for x in mapper[k]]))
    else:
        to_treat = mapper
        
    df.reset_index(drop = True , inplace= True)    
    
    condition = [x in to_treat for x in df.index.tolist()]
    df['outliers'] = np.where(condition , 1 , 0)
    
    return df

def outliers_processing(data ,type_locate='dbscan' , treat_or_delete='treat',\
                        columns = ['ead','MT_INI_FIN_','mt_appo_'] ,):
    """
    Cette fonction permet d'appliquer le traitement des outliers complet
    en utilisant les fonctions prédéfinies en haut

    """
    if type_locate =='dbscan':
        mapper = locate_outliers_dbscan(data, columns = columns )
    else:
        mapper = locate_outliers_zscore(data, columns = columns )
    
    if treat_or_delete == 'treat':
        data = treat_outliers(data , mapper )
    else:
        data = delete_outliers(data , mapper )
    
    return data


def concat_dicos(liste):
    combined = {}
    for k in liste[0].keys():
        combined[k] = [d[k] for d in liste]
    return combined




def tweak_maturity_lost_contracts (df):
    """
    Cette fonction permet d'appliquer un changement sur les maturités de certains contrats pour permettre à la fonction
    split_maturity de prendre l'intervalle entre 2 maturités.

    """
    max_mat = pd.DataFrame(df.groupby('cle2')['_maturity_'].max()).reset_index()
    jogo = { i : max_mat[max_mat['_maturity_']==i]['cle2'].tolist() for i in list(range(1,6))+[7,8,10,11,19,20,21,22,23]+list(range(13,18))}

    df['tweaked_maturity'] = df['_maturity_'].tolist()

    for i in jogo.keys():
        if i<6:
            snk=6
        elif i in list(range(13,18)):
            snk=18
        elif i in [7,8]:
            snk=9
        elif i in list(range(19,24)):
            snk=24
        elif i in [10,11]:
            snk=12
            
        df.loc[((df['cle2'].isin(jogo[i])) & (df['_maturity_']==i) ) , 'tweaked_maturity']   =   snk
          
    return df


def add_temporality(mat6 , mat9 , mat12 , mat18 , mat21)   : 
    mat9['tx_rec_marg_t_1'] = mat9['tx_rec_marg'].tolist()
    for k in mat9['cle2'].tolist(): 
        mat9.loc[mat9['cle2']==k , 'tx_rec_marg_t_1'] = mat6.loc[mat6['cle2'] == k,'tx_rec_marg'].tolist()[0]
        
    
    
    mat12['tx_rec_marg_t_1'] = mat12['tx_rec_marg'].tolist()
    for k in mat12['cle2'].tolist(): 
        mat12.loc[mat12['cle2']==k , 'tx_rec_marg_t_1'] = mat9.loc[mat9['cle2'] == k,'tx_rec_marg'].tolist()[0]
       
        
    mat18['tx_rec_marg_t_1'] = mat18['tx_rec_marg'].tolist()
    for k in mat18['cle2'].tolist(): 
        mat18.loc[mat18['cle2']==k , 'tx_rec_marg_t_1'] = mat12.loc[mat12['cle2'] == k,'tx_rec_marg'].tolist()[0]
       
    
    mat21['tx_rec_marg_t_1'] = mat21['tx_rec_marg'].tolist()
    for k in mat21['cle2'].tolist(): 
        mat21.loc[mat21['cle2']==k , 'tx_rec_marg_t_1'] = mat18.loc[mat18['cle2'] == k,'tx_rec_marg'].tolist()[0]
       
    
    return mat9 , mat12 , mat18 , mat21

def project_censored_contracts(df , mat24 ):
    """
    Cette fonction permet de créer des valeurs de tx_rec_marg à maturité 24 pour les
    contrats entrés en défaut 08 et 09 / 2015.
    Elle garde les caractéristiques inchangés du contrat, modifie uniquement la maturité 
    en 24 et rajoute la valeur du tx_rec_marg par la moyenne groupée par cat_seg.
    
    Tous les contrats censurés (118) sont rajoutés dans la base mat21.
    
    """
    df2 = df.copy()
    df2['dte_def_2'] =  pd.to_datetime(df2['dte_def_2'], format='%d/%m/%Y')


    ContProj=df2[(df2["dte_def_2"] > datetime(2015, 4, 30)) & \
                 (df2["dte_def_2"] < datetime(2015, 11, 30)) &\
                 (df2['_clos_']==0)]
    t = pd.DataFrame(ContProj.groupby('cle2')['_maturity_'].max())
        
    to_project = { k : t[t['_maturity_']==k].index.tolist() for k in range(19,24)}
      
    
    
    # dico18 = {}
        
    # for k in to_project[18] :
    #     x = df2[df2['cle2']==k]['tx_rec_act_2'].values
    #     x_ = [x[i] - x[i-1] for i in range(1,len(x))]
    #     model = AutoReg(x_, lags=1)
    #     model_fit = model.fit()
    #     yhat = model_fit.predict(len(x_), len(x_))
        
    #     x_.append(yhat[0])
    #     model = AutoReg(x_, lags=1)
    #     model_fit = model.fit()
    #     yhat = model_fit.predict(len(x_), len(x_)) 
        
    #     x_.append(yhat[0])
    #     model = AutoReg(x_, lags=1)
    #     model_fit = model.fit()
    #     yhat = model_fit.predict(len(x_), len(x_)) 
        
    #     x_.append(yhat[0])
    #     model = AutoReg(x_, lags=1)
    #     model_fit = model.fit()
    #     yhat = model_fit.predict(len(x_), len(x_)) 
        
    #     x_.append(yhat[0])
    #     model = AutoReg(x_, lags=1)
    #     model_fit = model.fit()
    #     yhat = model_fit.predict(len(x_), len(x_)) 
        
    #     x_.append(yhat[0])
    #     model = AutoReg(x_, lags=1)
    #     model_fit = model.fit()
    #     yhat = model_fit.predict(len(x_), len(x_)) 
        
    #     if w :=  yhat[0] + x_[-1] + x_[-2] + x_[-3] + x_[-4] + x_[-5]>=0:
    #         dico18[k] = w
    #     else:
    #         dico18[k] = 0
      
    
    dico19 = {}
        
    for k in to_project[19] :
        x = df2[df2['cle2']==k]['tx_rec_act_2'].values
        x_ = [x[i] - x[i-1] for i in range(1,len(x))]
        model = AutoReg(x_, lags=2)
        model_fit = model.fit()
        yhat = model_fit.predict(len(x_), len(x_))
        
        x_.append(yhat[0])
        model = AutoReg(x_, lags=2)
        model_fit = model.fit()
        yhat = model_fit.predict(len(x_), len(x_)) 
        
        x_.append(yhat[0])
        model = AutoReg(x_, lags=2)
        model_fit = model.fit()
        yhat = model_fit.predict(len(x_), len(x_)) 
        
        x_.append(yhat[0])
        model = AutoReg(x_, lags=2)
        model_fit = model.fit()
        yhat = model_fit.predict(len(x_), len(x_)) 
        
        x_.append(yhat[0])
        model = AutoReg(x_, lags=2)
        model_fit = model.fit()
        yhat = model_fit.predict(len(x_), len(x_)) 
        
        if  yhat[0] + x_[-1] + x_[-2] + x_[-3] + x_[-4] + x_[-5]>=0:
            dico19[k] =yhat[0] + x_[-1] + x_[-2] + x_[-3] + x_[-4] + x_[-5]
        else:
            dico19[k] = 0
    
    dico20 = {}
        
    for k in to_project[20] :
        x = df2[df2['cle2']==k]['tx_rec_act_2'].values
        x_ = [x[i] - x[i-1] for i in range(1,len(x))]
        model = AutoReg(x_, lags=2)
        model_fit = model.fit()
        yhat = model_fit.predict(len(x_), len(x_))
        
        x_.append(yhat[0])
        model = AutoReg(x_, lags=2)
        model_fit = model.fit()
        yhat = model_fit.predict(len(x_), len(x_)) 
        
        x_.append(yhat[0])
        model = AutoReg(x_, lags=2)
        model_fit = model.fit()
        yhat = model_fit.predict(len(x_), len(x_)) 
        
        x_.append(yhat[0])
        model = AutoReg(x_, lags=2)
        model_fit = model.fit()
        yhat = model_fit.predict(len(x_), len(x_)) 
        
        if yhat[0] + x_[-1] + x_[-2] + x_[-3] + x_[-4] + x_[-5]>=0:
            dico20[k] = yhat[0] + x_[-1] + x_[-2] + x_[-3] + x_[-4] + x_[-5]
        else:
            dico20[k] = 0
    
    dico21 = {}
        
    for k in to_project[21] :
        x = df2[df2['cle2']==k]['tx_rec_act_2'].values
        x_ = [x[i] - x[i-1] for i in range(1,len(x))]
        model = AutoReg(x_, lags=1)
        model_fit = model.fit()
        yhat = model_fit.predict(len(x_), len(x_))
        
        x_.append(yhat[0])
        model = AutoReg(x_, lags=1)
        model_fit = model.fit()
        yhat = model_fit.predict(len(x_), len(x_)) 
        
        x_.append(yhat[0])
        model = AutoReg(x_, lags=1)
        model_fit = model.fit()
        yhat = model_fit.predict(len(x_), len(x_)) 
        
        if   yhat[0] + x_[-1] + x_[-2] + x_[-3] + x_[-4] + x_[-5]>=0:
            dico21[k] = yhat[0] + x_[-1] + x_[-2] + x_[-3] + x_[-4] + x_[-5]
        else:
            dico21[k] = 0
    
    dico22 = {}
        
    for k in to_project[22] :
        x = df2[df2['cle2']==k]['tx_rec_act_2'].values
        x_ = [x[i] - x[i-1] for i in range(1,len(x))]
        model = AutoReg(x_, lags=2)
        model_fit = model.fit()
        yhat = model_fit.predict(len(x_), len(x_))
        
        x_.append(yhat[0])
        model = AutoReg(x_, lags=2)
        model_fit = model.fit()
        yhat = model_fit.predict(len(x_), len(x_)) 
        
        if   yhat[0] + x_[-1] + x_[-2] + x_[-3] + x_[-4] + x_[-5]>=0:
            dico22[k] = yhat[0] + x_[-1] + x_[-2] + x_[-3] + x_[-4] + x_[-5]
        else:
            dico22[k] = 0
    
    
    
    dico23 = {}
    
    for k in to_project[23] :
        x = df2[df2['cle2']==k]['tx_rec_act_2'].values
        x_ = [x[i] - x[i-1] for i in range(1,len(x))]
        model = AutoReg(x_, lags=2)
        model_fit = model.fit()
        yhat = model_fit.predict(len(x_), len(x_))
        
        if  yhat[0] + x_[-1] + x_[-2] + x_[-3] + x_[-4] + x_[-5]>=0:
            dico23[k] = yhat[0] + x_[-1] + x_[-2] + x_[-3] + x_[-4] + x_[-5]
        else:
            dico23[k] = 0
            
    dicos = {**dico23 , **dico22 , **dico21 , **dico20 , **dico19 }  
    
    
    
    inter = [to_project[k] for k in to_project.keys()]
    all_to_project =  [item for sublist in inter for item in sublist]
    #all_to_project = list(dicos.keys())
    lines_to_append = []
    for c in all_to_project:
        dict_0 = mat24[mat24['cle2']== c].reset_index(drop=True).transpose().to_dict()[0]
        dict_0['tx_rec_marg'] = dicos[c]
        dict_0['tweaked_maturity'] = 24
        lines_to_append.append(dict_0)
        
    to_concat = pd.DataFrame(concat_dicos(lines_to_append))
    
    mat24 = pd.concat([mat24 , to_concat])
    
    return mat24
    


def correct_projection(mat6 , mat9 , mat12 , mat18 , mat21):


    sums_tx = { k : mat21[mat21['cle2']==k]['tx_rec_marg'].tolist()[0] + mat18[mat18['cle2']==k]['tx_rec_marg'].tolist()[0] + \
        mat12[mat12['cle2']==k]['tx_rec_marg'].tolist()[0] + mat9[mat9['cle2']==k]['tx_rec_marg'].tolist()[0] + \
        mat6[mat6['cle2']==k]['tx_rec_marg'].tolist()[0] for k in mat21['cle2'].tolist() }
    
    for k in sums_tx.keys():
        if sums_tx[k] >1:
            mat21.loc[mat21['cle2']==k,'tx_rec_marg'] = 1 - ( mat18[mat18['cle2']==k]['tx_rec_marg'].tolist()[0] + \
                    mat12[mat12['cle2']==k]['tx_rec_marg'].tolist()[0] + mat9[mat9['cle2']==k]['tx_rec_marg'].tolist()[0] + \
                    mat6[mat6['cle2']==k]['tx_rec_marg'].tolist()[0] )
    
    return mat21


def split_maturity(df):
    """
    Cette fonction permet de : 1. Diviser le DataFrame par maturité
                               2. Créer le taux de recouvrement marginal tx_rec_marg

    Returns
    -------
    mat6 : df

    mat9 : df

    mat12 : df

    mat18 : df

    mat21 : df

    """
    mat0  = df[df['tweaked_maturity']==0].sort_values(by='cle2')
    mat6  = df[df['tweaked_maturity']==6].sort_values(by='cle2')
    mat9  = df[df['tweaked_maturity']==9].sort_values(by='cle2')
    mat12 = df[df['tweaked_maturity']==12].sort_values(by='cle2')
    mat18 = df[df['tweaked_maturity']==18].sort_values(by='cle2')
    mat21 = df[df['tweaked_maturity']==24].sort_values(by='cle2')
    
    mat0     = mat0[mat0['cle2'].apply(lambda x : x in mat6['cle2'].tolist())].sort_values(by='cle2')
    mat6_9   = mat6[mat6['cle2'].apply(lambda x : x in mat9['cle2'].tolist())].sort_values(by='cle2')
    mat9_12  = mat9[mat9['cle2'].apply(lambda x : x in mat12['cle2'].tolist())].sort_values(by='cle2')
    mat12_18 = mat12[mat12['cle2'].apply(lambda x : x in mat18['cle2'].tolist())].sort_values(by='cle2')
    mat18_21 = mat18[mat18['cle2'].apply(lambda x : x in mat21['cle2'].tolist())].sort_values(by='cle2')
    
    
    mat6['tx_rec_marg']  = mat6['tx_rec_act_2'].values  - mat0['tx_rec_act_2'].values
    mat9['tx_rec_marg']  = mat9['tx_rec_act_2'].values  - mat6_9['tx_rec_act_2'].values
    mat12['tx_rec_marg'] = mat12['tx_rec_act_2'].values - mat9_12['tx_rec_act_2'].values
    mat18['tx_rec_marg'] = mat18['tx_rec_act_2'].values - mat12_18['tx_rec_act_2'].values
    mat21['tx_rec_marg'] = mat21['tx_rec_act_2'].values - mat18_21['tx_rec_act_2'].values
    
    
    return mat6 , mat9 , mat12 , mat18 , mat21





def export_bases_df_maturity(chemin_DRIM, df, concat ,  mat6 , mat9 , mat12 , mat18 , mat21):
    
    """
    Cette fonction permet d'exporter les bases nettoyées, splités par maturités
    
    """
    
    margs = {6:mat6,9:mat9,12:mat12,18:mat18,24:mat21}
    
    for k in margs.keys():
        margs[k].to_csv(r'{}\Output\mat{}.csv'.format(chemin_DRIM,k),header=True,sep=';',index=False)
    
    df.to_csv(r'{}\Output\df_cleaned.csv'.format(chemin_DRIM),header=True,sep=';',index=False)
    concat.to_csv(r'{}\Output\concat_mats_cleaned.csv'.format(chemin_DRIM),header=True,sep=';',index=False)



















