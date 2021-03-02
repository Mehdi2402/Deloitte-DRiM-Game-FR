


Le script main_DRIM.py permet d'effectuer toutes les étapes de cleaning pour préparer les bases
	à la modélisation. Les fonctions de traitement sont appelées à partir du script cleaning.py
	-> Bases nettoyées


Les script selection_x.py permettent d'appliquer la sélection de variable et la standardisation
	pour préparer les bases au modèles qu'on appliquera sur SAS et python.
	-> Bases modélisation


Le script modelisation.sas permet d'appliquer la modélisation principale sur SAS et exporter les résultats.
	-> Sorties SAS


Les scripts rf.py et xgboost.py appliquent les modèles challenger et sortent les résultats.

Le script streamlit_structure.py permet de créer l'interface graphique (Webapp).