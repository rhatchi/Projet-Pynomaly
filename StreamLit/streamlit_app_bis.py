# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pandas as pd
import numpy as np
from Evaluations import *

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

import Conv_final_V2 as CF
import Dense_final_V2 as DF

from sklearn.metrics import classification_report, roc_auc_score
from joblib import load

########################
### Useful variables ###
########################

machine_types = ["fan","pump","valve","slider","ToyCar","ToyConveyor"]

IDs = {"fan":[0,2,4,6],
       "pump":[0,2,4,6],
       "valve":[0,2,4,6],
       "slider":[0,2,4,6],
       "ToyCar":[1,2,3,4],
       "ToyConveyor":[1,2,3]}

pages = ["Présentation du projet","Exploration du dataset","Modèles","Preprocessing","Application","Évaluation des modèles","Conclusion"]

df = pd.read_csv('/Users/romeo/FormationDataScientist/ProjetPynomaly/Path_DF.csv')


###############
### Sidebar ###
###############

st.sidebar.title('Pynomaly')

st.sidebar.subheader('Menu')

active_page = st.sidebar.radio("Choisissez une page",pages)

st.sidebar.info(
        "Projet DS - Promotion Bootcamp Avril 2021"
        "\n\n"
        "Participants:"
        "\n\n"
        "Roméo Hatchi (https://www.linkedin.com/in/rom%C3%A9o-hatchi-1aa226119/)"
        "\n\n"
        "Nicolas Gislais (https://www.linkedin.com/in/nicolas-gislais-947a6920a/) "
        )



#################################
# Page "Présentation du projet" #
#################################

if active_page == pages[0]:
    st.title("Projet Pynomaly")
    
    st.write('Le dataframe du projet contenant toutes les informations essentielles :')
    st.write(df)  


#################################
# Page "Exploration du dataset" #
#################################

if active_page == pages[1]:
    st.title("Dataset")
    
    fig = plt.figure()
    sns.countplot(df['Machine_Type'])
    plt.title('Nombre de prises de son par type de machine')
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(data=df,x='Machine_Type',hue='Dataset', palette="pastel")
    plt.title('Répartition des prises de son par type de machine et par Dataset')
    plt.xlabel('Type de machine')
    plt.ylabel('Nombre de prises de son')
    st.pyplot(fig)
    
    fig = plt.figure()
    sns.countplot(data=df,
            x='Machine_Type',
            hue='Machine_ID',
            #row='Dataset',
            palette="pastel")
    plt.title('Répartition des ID de machines par type de machine')
    plt.xlabel('Type de machine')
    plt.ylabel('Nombre de prises de son')    
    st.pyplot(fig)

    
    fig = plt.figure()
    sns.countplot(data=df[df['Dataset']=='test'],
            x='Machine_Type',
            hue='Status')
    plt.title('Répartition des labels cible par type de machine au sein des Datasets de test')
    plt.xlabel('Type de machine')
    plt.ylabel('Nombre de prises de son')
    st.pyplot(fig)


##################
# Page "Modèles" #
##################

if active_page == pages[2]:
    st.title("Modèles de deep-learning utilisés")
    
    st.subheader('Modèle Dense Auto-encoder :')
    FeatureDense= plt.imread("ModelDense.png")   
    st.image(FeatureDense)

    st.write('\n\n')
    st.subheader('Modèle Convolutif Auto-encoder :')    
    FeatureConv= plt.imread("ModelConv.png")   
    st.image(FeatureConv)


########################
# Page "Preprocessing" #
########################

if active_page == pages[3]:
    st.title(active_page)
    
    st.subheader('Sélection de features pour le modèle Dense Auto-encoder :')
    FeatureDense= plt.imread("FeatureDense.jpg")   
    st.image(FeatureDense)

    st.write('\n\n')
    st.subheader('Sélection de features pour le modèle Convolutif Auto-encoder :')    
    FeatureConv= plt.imread("FeatureConvolutif.jpg")   
    st.image(FeatureConv)
    
    st.write('source : Deep Dense and Convolutional Autoencoders for Unsupervised Anomaly Detection in Machine Condition Sounds')

    


######################
# Page "Application" #
######################

if active_page == pages[4]:
    st.title("Exemple d'application")
    
    model = st.radio("Choisissez un modèle auto-encoder", ['dense', 'convolutif'])

    machinetype = st.radio("Choisissez un type de machine",machine_types)
    
    machineID = st.radio("Choisissez l'ID d'une machine",IDs[machinetype])
    
    dataset = st.radio("Choisissez le dataset",['train','test'])
    
    df = df[(df.Dataset == dataset) & (df.Machine_Type == machinetype) & (df.Machine_ID == machineID) ]
    df = df.sample(n=100)
    st.write(df)

    if model == 'dense':
        model_trained = tf.keras.models.load_model('trained_models/denseAE_' + machinetype + '.hdf5', compile=False)
        X_true = DF.dataset_stream(df)
        X_pred = model_trained.predict(X_true)
        if machinetype == 'ToyCar': 
            error = DF.errors(X_true, X_pred, len(df), 588)
        else : 
            error = DF.errors(X_true, X_pred, len(df))
        
    else :
        scaler = load('scaler/std_scaler_'+ machinetype +'.bin')
        model_trained = tf.keras.models.load_model('trained_models/convAE_' + machinetype + '.hdf5', compile=False)
        X_true = CF.dataset_stream(df, scaler)
        X_pred = model_trained.predict(X_true)
        if machinetype == 'ToyCar': 
            error = CF.errors(X_true, X_pred, len(df), 105)
        else : 
            error = CF.errors(X_true, X_pred, len(df))
            
    y_true = df.Status.replace(['normal', 'anomaly'], [0,1])    
    
    # tracé de la courbe ROC
    fig = plt.figure()
    fpr, tpr, seuils = roc_curve(y_true, error, pos_label = 1)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, 'orange', label = 'Modèle dense (auc = %0.2f)' % roc_auc)
    plt.title('Courbe ROC')
    plt.xlabel('Taux faux positifs')
    plt.ylabel('Taux vrais positifs')
    plt.xlim(0,1)
    plt.ylim(0,1.05)

    plt.plot(fpr, fpr, 'b--', label = 'Aléatoire (aux = 0.5)')

    plt.legend(loc = 'lower right')
    st.pyplot(fig)
    
    auc = roc_auc_score(y_true, error)
    p_auc = roc_auc_score(y_true, error, max_fpr=0.1)
    st.write("L'AUC sur cet échantillon est :", auc)
    st.write("La pAUC sur cet échantillon est :", p_auc)
    
    seuil = st.number_input('Choisissez un seuil :')
    if seuil :
        y_pred = np.where(error[:] > seuil, 1, 0)
        #st.write(DF.matrice_confusion(y_true, y_pred))
        #st.write('\n\n')
        #st.write(classification_report(y_true, y_pred))
        
        st.write(DF.matrice_confusion(y_true, y_pred),'\n\n',classification_report(y_true, y_pred))

    



#################################
# Page "Évaluation des modèles" #
#################################

if active_page == pages[5]:
    st.title(active_page)
    
    st.subheader('Données d\'évaluation par ID')
                 
    infos = """errors_mean : erreur moyenne sur le dataset d'entraînement.
            \nerrors_std : écart-type des erreurs sur le dataset d'entraînement.
            \nAUC : aire sous la courbe ROC.
            \npAUC : aire partielle sous la courbe ROC qui mets plus d'importance à ce que le modèle ne fasse pas de faux positifs.
            \npercentile : percentile de seuil, déterminé pour maximiser la somme des F1-scores sur le dataset de test.
            \nthreshold : seuil d'erreur associé au percentile des erreurs du dataset d'entraînment
            \nTN : True Negative, nombre de prises de son bien labélisées 'normal'
            \nFN : False Negative, nombre de prises de son mal labélisées 'normal'
            \nFP : False Positive, nombre de prises de son mal labélisées 'anomaly'
            \nTP : True Positive, nombre de prises de son bien labélisées 'anomaly'
            \nprecision : précision de prédiction de la classe 'anomaly'. Le calcul pour l'obtenir est : TP/(TP+FP)."""
    show_infos = st.checkbox("Détails des variables")
    if show_infos:
        st.info(infos)
    
    st.write('Voici les résultats obtenus par ID de machines pour le modèle Dense')
    
    
    df = pd.DataFrame(columns = ["errors_mean","errors_std",
                                 "AUC","pAUC",
                                 "percentile","threshold",
                                 "TN","FN","FP","TP",
                                 "precision"])
    for machinetype in machine_types:
        evaluation_data_file = "results/"+machinetype+"/"
        for machineID in IDs[machinetype]:
            array = np.load(evaluation_data_file+"EvalData_DenseAE_"+machinetype+'_'+str(machineID)+".npy")
            df.loc[machinetype+'_'+str(machineID)]=array
    
    st.write(df)
        
        
    st.write('Voici les résultats obtenus par ID de machines pour le modèle Convoulutif')
    
    
    df2 = pd.DataFrame(columns = ["errors_mean","errors_std",
                                 "AUC","pAUC",
                                 "percentile","threshold",
                                 "TN","FN","FP","TP",
                                 "precision"])
    for machinetype in machine_types:
        evaluation_data_file = "results/"+machinetype+"/"
        for machineID in IDs[machinetype]:
            array = np.load(evaluation_data_file+"EvalData_ConvAE_"+machinetype+'_'+str(machineID)+".npy")
            df2.loc[machinetype+'_'+str(machineID)]=array
    
    st.write(df2)
    
    st.subheader('Comparaison des scores entre les 2 modèles')
    st.write("La comparaison est faite sur 3 métriques : l'AUC, la pAUC et la présicion associée au seuil maximisant la somme des F1-scores.")
    df3 = pd.DataFrame(index=df.index, columns = ["Dense_AUC","Conv_AUC",
                                  "Dense_pAUC","Conv_pAUC",
                                 "Dense_precision","Conv_precision"])
    df3["Dense_AUC"]=df["AUC"]
    df3["Dense_pAUC"]=df["pAUC"]
    df3["Dense_precision"]=df["precision"]
    df3["Conv_AUC"]=df2["AUC"]
    df3["Conv_pAUC"]=df2["pAUC"]
    df3["Conv_precision"]=df2["precision"]
    
    st.write(df3)


#####################
# Page "Conclusion" #
#####################

if active_page == pages[6]:
    st.title(active_page)



