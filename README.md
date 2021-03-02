# Deloitte DRiM Game - Risque de crédit

# Sujet
**Lien du challenge : https://www.deloitterecrute.fr/2020/12/23/drim-game-2020-retour-sur-le-challenge-data-science/**

Analyse des facteurs explicatifs (déterminants) des taux de récupérations marginales par tranche de maturité en défaut sur un portefeuille de crédits.

L’analyse, concernant des données de panel avec une dimension temporelle, est à traiter par application de modèles économétriques « classiques » et l’utilisation raisonnée d’une ou plusieurs méthodes de machine learning.

# Contexte

Identifier les facteurs influençant le taux de perte en cas de défaut (LGD)
 
Une contrainte réglementaire à satisfaire pour être autorisé à utiliser ces modèles pour le calcul des exigences en fonds propres

Un enjeu opérationnel pour optimiser le processus de recouvrement en orientant le « timing » et les modalités de actions de recouvrement

# Etapes

* Application de plusieus étapes de nettoyage et mise en forme des données.
* Rajout de variables macroéconomiques (Taux d'intérêt à la consommation, CPI, Taux de chômage, CLI (proxy mensuelle du PIB)).
* Rajout de variables de croisement, ratios et variables indicatrices.
* Séparation du dataset de maturités.
* Projection des contrats censurés.
* Prétraitement pour la modélisation : Discrétisation de la target et séléction de variables.
* Modélisation avec un **modèle classique** : régression logistique, **des modèles challenger** : Random Forest et XGBoost, et **un modèle alternatif** : cubist (régression bornée).

# Résultats
Réstitution des résultats dans une interface graphique utilisant **Streamlit** ( programme : *streamlit_structure.py*).
Aperçu des résultats :
<p align="center">
  <img src="https://github.com/Mehdi2402/images/blob/main/drim_contrat.gif?raw=true" />
</p>
<p align="center">
  <img src="https://github.com/Mehdi2402/images/blob/main/drim_mat.gif?raw=true" />
</p>

# Le code

Le script **main_DRIM.py** permet d'effectuer toutes les étapes de cleaning pour préparer les bases
	à la modélisation. Les fonctions de traitement sont appelées à partir du script **cleaning.py**
	-> Bases nettoyées


Les script **selection_x.py** permettent d'appliquer la sélection de variable et la standardisation
	pour préparer les bases au modèles qu'on appliquera sur SAS et python.
	-> Bases modélisation


Le script **modelisation.sas** permet d'appliquer la modélisation principale sur SAS et exporter les résultats.
	-> Sorties SAS


Les scripts **rf.py** et **xgboost.py** appliquent les modèles challenger et sortent les résultats.

Le script **streamlit_structure.py** permet de créer l'interface graphique (Webapp).
