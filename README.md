# Projet d'Analyse et Génération de Partitions de Musique Choro : 

## Description
Ce projet, mené dans le cadre de notre chef-d'œuvre de fin d'études de Master 2, se consacre à l'étude approfondie et à la génération automatisée de partitions de musique Choro, un genre musical brésilien riche et complexe. Notre objectif est double : d'abord, analyser un corpus de partitions de Choro pour en dégager les tendances mélodiques et harmoniques ; ensuite, utiliser ces informations pour entraîner des modèles de Machine Learning capables de générer de nouvelles compositions respectant les caractéristiques stylistiques du genre.

Pour ce faire, nous avons mis en œuvre trois types de modèles : Modèles LSTM (Long Short-Term Memory), Modèles de Markov et Modèles Transformer. 

## Usage
  ### show_melody.ipynb
  Permet de visualiser les mélodies générées presentes dans "Generation/Generated"
  ### extract_data_interface.ipynb
  Permet d'utiliser les fonctions pour extraire les données des fichiers xml et l'extraction de melodies pour analyser les données
  ### /Analyse
  * **analyse_chords.ipynb**  : Notebook d'aide pour la réparation future des données à la main. 
  * **dataset_stats.ipynb** : Notebook contenant toute l'analyse du dataset : signature temporelle, répartition des clés, pourcentage de chaque note, répartition des accords et expressions. 
  * **graph-stats.ipynb** : Notebook contenant les graphes de transition de notes des différentes parties des partitions du dataset. 
  ### /Generation
  * **Markov.ipynb**  : Notebook contenant toute la pipeline : traitement données, création du dataset , création du modele Markov , entrainement , génération et visualisation
  * **Transformer.ipynb** : Notebook contenant toute la pipeline : traitement données, création du dataset , création du modele Transformer , entrainement , génération et visualisation
  * **LSTM.ipynb** : Notebook contenant toute la pipeline : traitement données, création du dataset , création du modele LSTM , entrainement , génération et visualisation
  
  
## Requirements
 ### /Data/data_xml
 Le dossier contenant les 171 partitions en fichier xml.
 ### Modules python
 * matplotlib
 * numpy
 * seaborn
 * networkx          
 * tensorflow                2.11.0
 * music21                   7.3.3

 Ces modules peuvent s'installer grâce à la commande `pip install -r requirement.txt`. 

 ### Application 
 * Musescore3+

## Crédits
Ce projet inclut des portions de code (Generation/Transformeur et Generation/Markov) provenant du programme original détenu par Valerio Velardo. Le copyright du programme original est détenu par Valerio Velardo et sous licence MIT. Pour plus d'informations sur le programme original, [consultez son dépôt](https://github.com/musikalkemist/generativemusicaicourse/tree/main).

