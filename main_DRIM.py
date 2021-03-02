# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:02:50 2020

@author: mehdi
"""

import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
from datetime import datetime
from functools import reduce
import os
import matplotlib.dates as mdates
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import *
from typing import Union
from statsmodels.tsa.ar_model import AutoReg
from tqdm import trange

# Ecrivez ICI le chemin ou vous avez mis le dossier DRIM après l'avoir décompréssé

chemin_DRIM = r'C:\Users\mehdi\Desktop\M2 MOSEF\Scoring\DRIM'
os.chdir(r'{}\Codes'.format(chemin_DRIM))


from cleaning import add_stat_features , parse_date_inter ,throw_very_censored,\
                    select_timeframe_inter , import_celan_macro_tables ,\
                    add_macro_features, split_maturity , export_bases_df_maturity,\
                    label_encode , outliers_processing, project_censored_contracts,\
                    tweak_maturity_lost_contracts , croiseur,  correct_projection ,\
                    add_crise, add_temporality, valeur_ref , anomalie, add_fusion , \
                        add_fusion_import, concat_dicos
                    

from plot_stab_temporelle import eval_risk, eval_amount , get_evol,\
                                plot_stability ,preprocess_for_stability_plot


from discretization import preprocess_for_discretization , calc_distance ,\
                            select_best_k , discretization_machine , discretization_k22,\
                            discretization_k_all , discretize 


# Importer la base
data = pd.read_csv(r'{}\Base\Table_LGD.txt'.format(chemin_DRIM) , sep='\s')

df = data.dropna()

# Selectionner les maturités qui nous intéressent
df = df[df['_maturity_']<= 24]


# Supprimer les contrats très censurés
df = throw_very_censored(df)

# Définir si on veut exporter les bases nettoyées ou non
export = True

# Rajouter des features STATS
df = add_stat_features(df)


# Importer des features MACRO
chom_spain, chom_euro_area,\
cli_spain, cli_euro_area,\
tic_spain, tic_euro_area,\
cpi_spain , cpi_euro_area = import_celan_macro_tables(chemin_DRIM)


# Rajouter les features MACRO
dfs = [chom_spain, chom_euro_area, cli_spain, cli_euro_area, tic_spain, tic_euro_area, cpi_spain, cpi_euro_area]
df = add_macro_features(df,dfs)


# Diviser par le seuil
df = valeur_ref(df)

# Rajout des contrats fusionnés
#df = add_fusion(df)


# Split en maturité et 
df = tweak_maturity_lost_contracts (df)
mat6 , mat9 , mat12 , mat18 , mat24 = split_maturity(df)


# Rajout des contrats fusionnés
fusion = pd.read_csv(r'{}\Output\fusion.csv'.format(chemin_DRIM) , sep=';')
fusion = fusion.drop_duplicates()

mat6 , mat9 , mat12 , mat18 , mat24 = add_fusion_import(mat6,fusion),add_fusion_import(mat9,fusion),\
    add_fusion_import(mat12,fusion) , add_fusion_import(mat18,fusion), add_fusion_import(mat24,fusion)


# Traitement des valeurs aberrantes
type_locate = "dbscan"     # dbscan , zscore  
treat_or_delete = "treat"  # treat , delete
columns = ['ead','MT_INI_FIN_','mt_appo_']  #liste de colonnes à checker pour valeurs aberrantes

mat6 , mat9 , mat12 , mat18 , mat24 = outliers_processing( mat6 ,  type_locate  , treat_or_delete , columns),\
            outliers_processing( mat9 ,  type_locate  , treat_or_delete , columns),\
            outliers_processing( mat12 ,  type_locate  , treat_or_delete , columns),\
            outliers_processing( mat18 ,  type_locate  , treat_or_delete , columns),\
            outliers_processing( mat24 ,  type_locate  , treat_or_delete , columns)


# Rajout de crise
mat6 , mat9 , mat12 , mat18 , mat24 = add_crise(mat6,chemin_DRIM) , add_crise(mat9,chemin_DRIM),\
    add_crise(mat12,chemin_DRIM) , add_crise(mat18,chemin_DRIM) , add_crise(mat24,chemin_DRIM)


# Projection des contrats censurés
# Paramètre optionnel 'ar' qui correspond au p de AR(p) par defaut p=2
# concat les 5 maturités
mat24 = project_censored_contracts(df , mat24 )
mat24 = correct_projection(mat6 , mat9 , mat12 , mat18 , mat24)



# Rajouter la temporalité
#mat9 , mat12 , mat18 , mat21 = add_temporality(mat6 , mat9 , mat12 , mat18 , mat21) 


# Label encoding puis croiser les variables
for col in ['cat_seg','CD_CAT_EXPO_4','qual_veh','nat_veh']:
    mat6, mat9 , mat12, mat18, mat24 = label_encode(mat6,col) , label_encode(mat9,col) , label_encode(mat12,col) ,\
                                       label_encode(mat18,col) , label_encode(mat24,col)
    
list_cat = ['cat_seg','CD_CAT_EXPO_4','qual_veh']
list_cont = ['MT_INI_FIN_','mt_appo_','DUR_PREV_FIN','ead']
mat6, mat9 , mat12, mat18, mat24 = croiseur(mat6, list_cat , list_cont)  , croiseur(mat9, list_cat , list_cont),\
                                   croiseur(mat12, list_cat , list_cont) , croiseur(mat18, list_cat , list_cont),\
                                   croiseur(mat24, list_cat , list_cont)    


# Choisir si on discrétise
# disc = False
# VarList=['ead','DUR_PREV_FIN','MT_INI_FIN_','dur_b_defm_', 'dur_b_endm_', 'mt_appo_','pct_appo_','ratio_b_endm_']
# drop_existing=False
# if disc:
#     # Choisir le type de discretisation ici
#     type_disc = 'k_all'  # k_all / k22 / gmm / 
#     mat6 , mat9 , mat12 , mat18 , mat21 , concat = discretize(mat6,type_disc,VarList,drop_existing),discretize(mat9,type_disc,VarList,drop_existing),\
#                                                    discretize(mat12,type_disc,VarList,drop_existing),discretize(mat18,type_disc,VarList,drop_existing),\
#                                                    discretize(mat21,type_disc,VarList,drop_existing)


# Exportation des bases
if export: 
    concat = pd.concat([mat6,mat9,mat12,mat18,mat24])
    export_bases_df_maturity(chemin_DRIM, df, concat, mat6 , mat9 , mat12 , mat18 , mat24)



# Plot de stabilité Temporelle :

stability_plot = False    

if stability_plot:
    mat6_plot = preprocess_for_stability_plot(mat6)
    mat21_plot = preprocess_for_stability_plot(mat21)
    #Créer la liste des variables que l'on veut plus
    Liste=['CD_CAT_EXPO_4', 'cat_seg', 'nat_veh', 'qual_veh']
    plot_stability(mat6_plot, Liste, "dtf_per_trt", "Tx_rec_marg_Bin", 1)
    plot_stability(mat21_plot, Liste, "dtf_per_trt", "Tx_rec_marg_Bin", 1)
    
