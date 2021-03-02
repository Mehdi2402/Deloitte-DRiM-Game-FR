# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 21:53:11 2020

@author: mehdi
"""


import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report
import statsmodels.api as sm
import sys
import io
import numpy as np
from statsmodels.tools import add_constant
from statsmodels.iolib.smpickle import load_pickle
import plotly.graph_objects as go
import pickle

path = r'C:\Users\mehdi\Desktop\M2 MOSEF\Scoring\DRIM'







def plot_p0_temporality(dfs,contrat):
    ListContrat=[]
    ListGlob=[]
    ListeProfil=[]
    ListMat=[]
    
    for x in ['maturité 6','maturité 9','maturité 12','maturité 18', "maturité 24"]:
        df = dfs[x]   
        ListContrat.append(list(df[df["cle2"].isin([contrat])]['P_0'])[0])
        ListeProfil.append(list(df[df["cle2"]==contrat]["risque"])[0])
        ListGlob.append(df["P_0"].mean())
        ListMat.append(x)
    
    
    
    d = {'Contrat sélectionné': ListContrat, 'Tous les contrats': ListGlob, "Risque":ListeProfil}
    
    Graphe=pd.DataFrame.from_dict(d)
    Graphe.index=ListMat
    
    fig = go.Figure()
    for x in Graphe.iloc[:,:-1].columns:
        fig.add_trace(go.Scatter(x=Graphe.index, y=Graphe[x],\
                        mode="markers+lines",name=x,text=Graphe["Risque"]))
    
    for x in [x for x in range(1,len(ListeProfil)) if ListeProfil[x] != ListeProfil[x-1]]:
        print(x)
        fig.add_annotation(x=x,\
                           y=ListContrat[x],
                    text="Changement profil de risque",
                    showarrow=True,
                    arrowhead=1)
    fig.update_layout(title='P_0 temporality plot',
                       xaxis_title='Maturity',
                       yaxis_title='Probability')
    
    return fig
    #fig.write_html(r"C:\Users\mehdi\Desktop\M2 MOSEF\Scoring\DRIM\Prez\file.html")

def import_models(i):
    model = load_pickle(r"{}\models\model{}.pickle".format(path,i))
    
    return model

def import_models_sk(i):
    model = pickle.load(open(r"{}\models\model{}sk.sav".format(path,i),'rb'))
    
    return model

def multi_model(df):
   
    cm = pd.crosstab(df["tx_rec_marg_Bin"], df["I_tx_rec_marg_Bin"])
    
    class_report=classification_report(df["tx_rec_marg_Bin"], df["I_tx_rec_marg_Bin"],output_dict=True)
    
    accuracy = class_report['accuracy']
    
    class_report = pd.DataFrame(class_report).drop('accuracy', axis=1).transpose()
    
    # col_ix = pd.MultiIndex.from_product([['Classes prédite'], list('012')]) 
    # row_ix = pd.MultiIndex.from_product([['Classe réelle'], list('012')])
    
    # cm = cm.set_index(row_ix)
    # cm.columns = col_ix
    
    
    # old_stdout = sys.stdout
    # new_stdout = io.StringIO()
    # sys.stdout = new_stdout
    
    # print(cm)
    
    # output = new_stdout.getvalue()
    
    # sys.stdout = old_stdout
    
    
    return cm , class_report , accuracy

def param_est_sas(mat):
    
    p = pd.read_excel(r'{}\sas_output\param_est_sas\log{}.xlsx'.format(path,mat))


    p = p.drop(columns=['_Proc_', '_Run_','ClassVal0','DF','_ESTTYPE_'])
    
    est1 = p[p['Response']==1].drop('Response',axis=1)
    est2 = p[p['Response']==2].drop('Response',axis=1)
    
    return est1 , est2

# def multi_model_test(df):

#     #pop = pd.get_dummies(pop,prefix=['cat_seg',"date_neg"], columns = ['cat_seg',"date_neg"],drop_first=True)
#     probss = [e for e in df.columns if e.startswith('P_')]
#     matxnodiscr = df.drop(columns = ['F_tx_rec_marg_Bin','I_tx_rec_marg_Bin','max_p','risque','cle2'] + probss)
    
#     X=matxnodiscr.drop("tx_rec_marg_Bin", axis=1)
    
#     y=matxnodiscr["tx_rec_marg_Bin"]
    
    
#     #X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size= 0.2 ,\random_state=42)
    
    
#     logit_model=sm.MNLogit(y,sm.add_constant(X))
    
#     result=logit_model.fit_regularized()
    
    
#     stats=result.summary()
#     results_as_html = stats.tables[1].as_html() 
    
#     test = pd.read_html(results_as_html, header=0, index_col=0)[0] # meilleur forme
    
#     test0 = test.reset_index().rename(columns = {'tx_rec_marg_Bin=1':'coefs'})
#     seperator = test0[test0['coefs']=='const'].index.tolist()[1]
    
#     test1 = test.iloc[:seperator-1,:]
#     test2 = test.iloc[seperator:,:]
#     test2.index = test2.index.rename('tx_rec_marg_Bin=2')
#     return test1 , test2, result


def get_predicted_probs(skr,trained_model):
    
    pop = skr.copy()
    probas_preds = trained_model.predict(add_constant(pop))
    
    for i in probas_preds.columns:
        probas_preds.rename(columns={i : 'P_' + str(i)},inplace=True)
        
    probas_preds['I_tx_rec_marg_Bin'] = probas_preds.idxmax(axis=1).apply(lambda x : int(x[-1]))
    
    pop[probas_preds.columns] = probas_preds
    
    pop = define_risk(pop)
    return pop

def get_predicted_probs_sk(skr,trained_model):
    
    pop = skr.copy()
    probas_preds = trained_model.predict_proba(pop)
    probas_preds = pd.DataFrame(probas_preds)
    for i in probas_preds.columns:
        probas_preds.rename(columns={i : 'P_' + str(i)},inplace=True)
        
    probas_preds['I_tx_rec_marg_Bin'] = probas_preds.idxmax(axis=1).apply(lambda x : int(x[-1]))
    
    pop[probas_preds.columns] = probas_preds
    
    pop = define_risk(pop)
    return pop

def define_risk (df ,  seuil = 0.5) :
    probs = df[['P_0','P_1','P_2']]
    max_p = probs.idxmax(axis=1)
    df['max_p'] = max_p
    df['risque'] = [str(i) for i in range(df.shape[0])]
    for i in range(df.shape[0]):

        if df.loc[i,'max_p']=='P_1':

            if df.loc[i,df.loc[i,'max_p']]>seuil :
                df.loc[i,'risque'] = 'risque moyen'

            else :
                to_arg_max = pd.DataFrame(df.loc[i,['P_0','P_2']]).transpose().astype('float')

                if to_arg_max.idxmax(axis=1).tolist()[0] == 'P_0' :
                    df.loc[i,'risque'] = 'risque moyen supérieur'

                elif to_arg_max.idxmax(axis=1).tolist()[0] == 'P_2' :
                    df.loc[i,'risque'] = 'risque moyen inférieur'

        elif df.loc[i,'max_p']=='P_2':
            df.loc[i,'risque'] = 'risque inférieur'
        else :
            df.loc[i,'risque'] = 'risque supérieur'


    return df

def distrib_plot(df):
    
    df=df[['tx_rec_marg_Bin', 'P_0']].copy()
    df['P_0'] = round(df['P_0']*100)

    (df['P_0'].value_counts(1)*100).sort_index().plot(kind = "bar", figsize = (22,7),label = "Train")
    #plt.xticks(np.arange(0, 100, step=5), np.arange(0, 100, step=5))
    plt.title("Graphe de points d'accumulation")
    plt.xlabel('Probabilité prédite')
    plt.ylabel('% de la population')
    fig= plt.show()
    return fig
    
def score_table(df):
    y_proba=df[["P_0"]]
    y_test=df['tx_rec_marg_Bin']
    predM1C = pd.concat([y_test, y_proba],axis=1)


    #Construction du tableau déciles
    Recapdf=pd.DataFrame()
    test=pd.qcut(predM1C['P_0'], 10).astype(str)
    intervalle=test.unique()
    str_intervals = [i.replace("(","").replace("]", "").split(", ") for i in intervalle]
    str_intervals=sorted(str_intervals, key=lambda x: x[0])
    test =pd.qcut(predM1C['P_0'], 10,labels=False)
    volume=test.value_counts()
    volume=volume.sort_index()
    data_crosstab = pd.crosstab(test,predM1C['tx_rec_marg_Bin'],margins = False)


    Risk_rate= data_crosstab[0]/volume
    Recapdf["Intervalles"]=str_intervals
    Recapdf["Volume"]=volume
    Recapdf = pd.concat([Recapdf, data_crosstab],axis=1)
    Recapdf["NoRefundRate"]=Risk_rate

    return Recapdf    
    
def decile_plot(Recapdf):
    plt.figure(figsize=(12,6))
    plt.bar(Recapdf.index.to_numpy(),Recapdf['NoRefundRate']*100)
    #plt.xticks(Recapdf.index.to_numpy(), predM11C['NoteScore'])
    plt.title('% de non-rembourseurs par decile de score', fontsize=30)
    plt.xlabel('Déciles', fontsize=20)
    plt.ylabel('Taux de non rembourseurs (en %)', fontsize=20)
    fig = plt.show()
    return fig

def load_data(math):
    df = pd.read_excel(r'{}\sas_output\echantillon\probas{}.xlsx'.format(path,math))
    
    mat = pd.read_csv(r'{}\Output11\echantillon\mat{}.csv'.format(path,math),sep=',')
    # mat = mat[mat['outliers']!=1]
    # mat = mat[mat['MT_INI_FIN_']!=0]
    # if math in ['12','9']:
    #     mat = mat[mat['tx_rec_marg']>=0]
    cle2 = mat['cle2'].tolist()      
    
    df['cle2'] = cle2 
    
    df = define_risk(df)
    df = df.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)
    return df 

var_cat ={6 : ['CD_CAT_EXPO_4','date_neg','fusion'],
          9 : ['CD_CAT_EXPO_4','cat_seg','qual_veh','fusion','date_neg'],
          12: ['date_neg','fusion'],
          18: ['qual_veh','fusion','date_neg','no_appo'],
          24: ['date_neg','CD_CAT_EXPO_4','fusion']}


signification = { 'cat_seg' : { 1 : 'Particulier' , 0 : 'Entreprises'},
                 'date_neg' : {1 : 'Restructuré' , 0 : 'Non Restructuré'},
                 'CD_CAT_EXPO_4' : {1 : 'Leasing', 0 :'Crédit'},
                 'no_appo' : { 1:"Pas d'apport", 0 : 'Apport positif'},
                 'qual_veh' : {1: "Véhicule occasion" , 0 : "Véhicule neuf"},
                 'fusion' : {1 : 'Fusionnés' , 0 : 'Non fusionnés'}}


def tableau_risk_classe(df,cat,signification):
    full_index = ["risque supérieur","risque moyen supérieur","risque moyen","risque moyen inférieur","risque inférieur"]
    
    risk_table = pd.DataFrame({'nan' : [np.nan]*5})
    risk_table.index = full_index
    
    class_risk = df['risque'].unique().tolist()
    
    glob_list = pd.DataFrame(df['risque'].value_counts(normalize=True)).rename(columns={'risque' : 'Global'})

    risk_table = risk_table.join(glob_list, how='left' , on = risk_table.index) 

    for var in cat :
        list_mod = df[var].unique().tolist()
        
        dict_columns_per_cat = {mod : pd.DataFrame(df[df[var]==mod]['risque'].value_counts(normalize=True))\
                                .rename(columns={'risque' : signification[var][mod]}) for mod in list_mod}
        
        
        for mod in list_mod:
            risk_table = risk_table.join(dict_columns_per_cat[mod], how='left' , on = risk_table.index) 
            
    
    risk_table.drop(columns=['nan'],inplace=True)


    return risk_table

def treatment (df, Pdigthresh = 0.5):
    df['DigitalProfil'] = np.random.uniform(0, high=1, size=len(df))
    
    df['treatment'] = np.where(df['risque']=="risque supérieur", \
                               'Appel téléphonique + Courrier','')
    df['treatment'] = np.where(df['risque']=="risque inférieur", \
                               'SMS',df['treatment'])

    df["treatment"] = np.where((df["DigitalProfil"] >= Pdigthresh) & \
                   ((df["risque"] == "risque moyen") | (df["risque"] == "risque moyen inférieur")), \
                       "Mail", df["treatment"])
    df["treatment"] = np.where((df["DigitalProfil"] < Pdigthresh) & \
                   ((df["risque"] == "risque moyen") | (df["risque"] == "risque moyen inférieur")), \
                       "Courrier", df["treatment"])
    df["treatment"] = np.where((df["DigitalProfil"] >= Pdigthresh) & \
                   (df["risque"] == "risque moyen supérieur"), \
                       "Appel téléphonique + mail", df["treatment"])
    df["treatment"] = np.where((df["DigitalProfil"] < Pdigthresh) & \
                   (df["risque"] == "risque moyen supérieur"), \
                       "Courrier", df["treatment"])
    return df

def resume_contrat(dfs, selected_contrat):
    predicted_class = [dfs[k][dfs[k]['cle2']==selected_contrat]['I_tx_rec_marg_Bin'].tolist()[0] for k in dfs.keys()] 
    predicted_profile = [dfs[k][dfs[k]['cle2']==selected_contrat]['risque'].tolist()[0] for k in dfs.keys()]
    predicted_treatment = [dfs[k][dfs[k]['cle2']==selected_contrat]['treatment'].tolist()[0] for k in dfs.keys()]
    summary_contrat = pd.DataFrame({'Classe prédite' : predicted_class , 'Profil de risque':predicted_profile,
                                    'Traitement' : predicted_treatment})
    
    summary_contrat.index = dfs.keys()
    
    return summary_contrat


def write_page_1(df,maturity):
    '''## Estimation des paramètres'''
    
    number = int(re.findall(r'\d+',maturity)[0])
    
    logit_coefs1 , logit_coefs2  = param_est_sas(number)
    
    st.markdown('**Classe du taux de recouvrement marginal = 1**')
    st.write(logit_coefs1)
    st.markdown('**Classe du taux de recouvrement marginal = 2**')
    st.write(logit_coefs2)
    
    '''## Matrice de confusion | Rapport de classification'''
    
    cm ,class_report , accuracy = multi_model(df)
    
    i1 , i2 , i3 = st.beta_columns([1,3,9])
    for i in range(5):
        i1.text('')
    i1.markdown('**Classes réelles**')
    i2.markdown('**Classes prédites**')
    i2.write(cm)
    #i3.markdown('...............................**Rapport de classification**...............................')
    i3.dataframe(class_report)
    i3.markdown('**accuracy : **' + str(round(accuracy,2)))    
    
    ''' ## Plots '''
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    distrib_plot(df)
    
    st.pyplot(distrib_plot(df))
    
    Recapdf = score_table(df)
        
    col1, col2 = st.beta_columns([3,3])
    
    col1.subheader("No Refund Data")
    col1.write(Recapdf[['Intervalles',0,'NoRefundRate']])
    
    col2.subheader("No Refund Plot")
    col2.pyplot(decile_plot(Recapdf))
    
    

    
    
def write_page_2(dfs):    
    '''# Brrrrrrrrrrrrrrrr'''
        
    data_file = st.file_uploader("Upload CSV or XLSX",type =["csv","xlsx"])
    
    if data_file :

        entry = pd.read_csv(data_file)
        
        st.markdown('**Dimensions du jeu de données :** ' + str(entry.shape))

        #st.dataframe(entry)
        
        
    
        
        '''# A l'échelle des maturités : '''
        selected_maturity = st.selectbox('Choisir une maturité :', ["maturité 6", "maturité 9","maturité 12","maturité 18","maturité 24"])
        
        
        '''## Règle de décison : Profils de risque'''
        image = Image.open(r'C:\Users\mehdi\Desktop\M2 MOSEF\Scoring\DRIM\Prez_\arbre_decision_profils.png')
        st.image(image, width=620, caption = 'Règles de construction des profils de risque')
        
        
        '''## Evolution des probabilités par classe de risque'''
        no = re.findall(r'\d+',selected_maturity)[0]
        image2 = Image.open(r'C:\Users\mehdi\Desktop\M2 MOSEF\Scoring\DRIM\bar_charts\BarChartMat{}.png'.format(no))
        st.image(image2, width=620)
        
        '''## Profils de risque par maturité segmentés : '''
        
        risk_table = tableau_risk_classe(dfs[selected_maturity],var_cat[int(re.findall(r'\d+',selected_maturity)[0])],signification)
        st.dataframe(risk_table)
        
        image3 = Image.open(r'C:\Users\mehdi\Desktop\M2 MOSEF\Scoring\DRIM\Prez_\trait.png')
        st.text('')
        st.image(image3, width=700)
        
        '''# A l'échelle d'un contrat  : '''          
        selected_contrat = st.selectbox('Select contrat :', dfs['maturité 24']["cle2"].tolist() )
        
        fig = plot_p0_temporality(dfs = dfs ,contrat= selected_contrat)
        st.plotly_chart(fig)
        
        dfs = { k : treatment(dfs[k]) for k in dfs.keys()}
        
        summary_contrat = resume_contrat(dfs,selected_contrat)
        st.dataframe(summary_contrat)
            
            
            
    
if __name__ =='__main__':
    
    page = st.sidebar.selectbox(label = "Mode : " , options=['Exploration','Opérationnel'])
    dfs = { 'maturité '+ str(k) : load_data(str(k)) for k in [6,9,12,18,24]}

    if page == 'Exploration':
        choices = ["maturité 6", "maturité 9","maturité 12","maturité 18","maturité 24"]
        
        '''## Choix de maturité '''
        
        maturity = st.selectbox(label = " Modèle : ", options = choices,key=12 )
        write_page_1(dfs[maturity],maturity)
    if page == 'Opérationnel':
        write_page_2(dfs)
    