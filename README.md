# Deloitte DRiM Game - Risque de crédit

# Sujet 
Analyse des facteurs explicatifs (déterminants) des taux de récupérations marginales par tranche de maturité en défaut sur un portefeuille de crédits.

L’analyse, concernant des données de panel avec une dimension temporelle, est à traiter par application de modèles économétriques « classiques » et l’utilisation raisonnée d’une ou plusieurs méthodes de machine learning.

# Contexte

Identifier les facteurs influençant le taux de perte en cas de défaut (LGD)
 
Une contrainte réglementaire à satisfaire pour être autorisé à utiliser ces modèles pour le calcul des exigences en fonds propres

Un enjeu opérationnel pour optimiser le processus de recouvrement en orientant le « timing » et les modalités de actions de recouvrement
![image](https://user-images.githubusercontent.com/56029953/109669041-04167180-7b72-11eb-8892-07f032594f65.png)

# Etapes

Application de plusieus étapes de nettoyage et mise en forme des données.
Rajout de variables macroéconomiques (Taux d'intérêt à la consommation, CPI, Taux de chômage, CLI (proxy mensuelle du PIB)).
Rajout de variables de croisement, ratios et variables indicatrices.
Séparation du dataset de maturités.
Projection des contrats censurés.
Prétraitement pour la modélisation : Discrétisation de la target.
Modélisation avec un **modèle classique** : régression logistique, **des modèles challenger** : Random Forest et XGBoost, et **un modèle alternatif** : cubist (régression bornée)

# Résultats

Réstitution des résultats dans une interface graphique utilisant **Streamlit** ( programme : *streamlit_structure.py*)
Aperçu des résultats
