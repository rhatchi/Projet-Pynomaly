import librosa
import sys
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU
import os
from sklearn.metrics import f1_score, roc_auc_score





#########################################
#
#             Preprocessing
#
#########################################



# Fonction de chargement d'un fichier audio
def file_load_stream(filepath, mono=False):
    frameSize = librosa.get_samplerate(filepath)
    hoplength = frameSize // 2
    stream = librosa.stream(filepath, 
                            block_length=1,
                            frame_length=frameSize,
                            hop_length=hoplength,
                            mono=mono)
    return stream

# Fonction qui fait une sélection de features MFEC sur un fichier audio et qui le transforme en array numpy
def file_to_vector_array_stream(filepath, n_mels=128, frames=5, n_fft=1024, hop_length=512, power=1):
    
    # Calcul des dimensions
    dims = n_mels * frames

    # Création du stream
    stream = file_load_stream(filepath)
    
    # Récupération de la fréquence d'échantillonnage
    sr = librosa.get_samplerate(filepath)
    
    liste = []
    for n, y in enumerate(stream):
        # Création du melspectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr,
                                                         n_fft=n_fft,
                                                         hop_length=hop_length, 
                                                         n_mels=n_mels, 
                                                         power=power)

        # Conversion en log-melspectrogram
        log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

        # Calcul de la taille du vecteur
        vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1  
        
        # Sortie pour les vecteurs "trop courts"
        if vector_array_size < 1:
            return np.empty((0, dims))

        # Création du vecteur de feature
        vector_array = np.zeros((vector_array_size, dims))

        for t in range(frames):
            vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

        liste.append(vector_array)

    liste = np.asarray(liste)
    # Redimensionnement
    liste = liste.reshape(liste.shape[0] * liste.shape[1], liste.shape[2])
    return liste

# Fonction de création du dataset
def dataset_stream(set_files):
    """
    renvoie un dataset sur laquelle entrainer/évaluer le modèle Dense_AE
    set_files est un DataFrame et sa première colonne contient les chemins
    """
    liste = []
    for k in range(len(set_files)):
        for l in file_to_vector_array_stream(set_files.iloc[k,0]):
            liste.append(l)
    return np.asarray(liste)




##########################
##### Recap function #####
##########################


### Pour l'entraînement du modèle ###
def feature_selection_train(path_csv, machinetype):
    # Chargement du dataframe des chemins d'accès
    df = pd.read_csv(path_csv)
    
    # Création du dataset d'entraînement
    train_data = df[(df.Dataset == "train") & (df.Machine_Type == machinetype)]
    train_data_stream = dataset_stream(train_data)
    
    return train_data_stream




#########################################
#
#             Model
#
#########################################


def model_denseAE(input_shape = (640,)):
    model_dense = Sequential()

    # Première couche Encoder
    model_dense.add(Dense(512, input_shape = input_shape))
    model_dense.add(BatchNormalization())
    model_dense.add(ReLU())

    # Seconde couche Encoder
    model_dense.add(Dense(512))
    model_dense.add(BatchNormalization())
    model_dense.add(ReLU())

    # Troisième couche Encoder
    model_dense.add(Dense(512))
    model_dense.add(BatchNormalization())
    model_dense.add(ReLU())

    # Quatrième couche Encoder
    model_dense.add(Dense(512))
    model_dense.add(BatchNormalization())
    model_dense.add(ReLU())

    # Couche goulot
    model_dense.add(Dense(8))
    model_dense.add(BatchNormalization())
    model_dense.add(ReLU())

    # Première couche Decoder
    model_dense.add(Dense(512))
    model_dense.add(BatchNormalization())
    model_dense.add(ReLU())

    # Seconde couche Decoder
    model_dense.add(Dense(512))
    model_dense.add(BatchNormalization())
    model_dense.add(ReLU())

    # Troisième couche Decoder
    model_dense.add(Dense(512))
    model_dense.add(BatchNormalization())
    model_dense.add(ReLU())

    # Quatrième couche Decoder
    model_dense.add(Dense(512))
    model_dense.add(BatchNormalization())
    model_dense.add(ReLU())

    # Couche de reconstruction 
    model_dense.add(Dense(640))
    
    return model_dense
    

##########################
##### Recap function #####
##########################

def model_dense_training(machinetype,train_data_stream):
    # Instanciation du modèle et compilation
    model_dense = model_denseAE()
    model_dense.compile(optimizer='adam', loss="mse")
    
    # Callback pour arrêter l'entrainement
    # et récupérer le meilleur modèle si la métrique ne diminue plus pendant 10 epochs
    early_stopping = callbacks.EarlyStopping(monitor = 'val_loss',
                                             patience = 10,
                                             mode = 'min',
                                             restore_best_weights = True)
    
    # Callback pour sauvegarder le meilleur modèle
    checkpoint = callbacks.ModelCheckpoint(filepath = os.getcwd()+'\\denseAE_'+machinetype+'.hdf5',
                                           monitor = 'val_loss', 
                                           save_best_only = True,
                                           mode = 'min')
    
    # Entraînement du modèle
    history = model_dense.fit(train_data_stream, train_data_stream,
                batch_size = 512,
                epochs = 100,
                callbacks=[checkpoint, early_stopping],
                validation_split = 0.3)
    
    # Sauvegarde des fonctions de perte au cours de l'entraînement
    np.save("HistoryValLoss_DenseAE_"+machinetype,np.asarray(history.history['val_loss']))
    np.save("HistoryLoss_DenseAE_"+machinetype,np.asarray(history.history['loss']))
    
    return model_dense





#########################################
#
#             Evaluation
#
#########################################
    
# Fonction de calcul des erreurs entre le jeu de départ et les prédictions du modèle
def errors(X_true, X_pred, length, nb_extract = 532): 
    """
    WARNING : nb_extract = 588 pour ToyCar
    """
    vect_error = np.mean(np.square(X_true - X_pred), axis=1)
    errors = np.zeros(length)
    for k in range(length):
        errors[k] = np.mean(vect_error[k*nb_extract : (k+1)*nb_extract])
    return errors

   

# Fonction calculant les F1-scores de chaque classe et la somme suivant le seuil choisi
# et retournant le precentile maximisant la somme
def f1scores(error_train_data,error_test_data,y_true,machinetype,machineID):
    f1_0 = []
    f1_1 = []
    for i in range(101):
        seuil = np.percentile(error_train_data,i)
        y_pred = np.where(error_test_data[:] > seuil, 1, 0)
        f1_1.append(f1_score(y_true, y_pred))
        f1_0.append(f1_score(y_true, y_pred,pos_label=0))
    
    # Calcul de la somme des scores
    somme = [x + y for x, y in zip(f1_1, f1_0)]
    
    # Sauvegarde des données
    np.save("DenseAE_F1_1"+machinetype+'_'+str(machineID),np.asarray(f1_1))
    np.save("DenseAE_F1_0"+machinetype+'_'+str(machineID),np.asarray(f1_0))
    np.save("DenseAE_F1_somme"+machinetype+'_'+str(machineID),np.asarray(somme))
    
    return np.where(somme==max(somme))[0][0]


# Fonction retournant un dataframe représentant la matrice de confusion
def matrice_confusion(y_true, y_pred):
    return pd.crosstab(y_true, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])

    

##########################
##### Recap function #####
##########################

def model_evaluation(path_csv, machinetype, model_dense):
    
    if machinetype == "ToyCar" : nb_extract = 588
    else : nb_extract = 532
        
    # Chargement du dataframe des chemins d'accès
    df = pd.read_csv(path_csv)
    
    IDs = {"fan":[0,2,4,6],
       "pump":[0,2,4,6],
       "valve":[0,2,4,6],
       "slider":[0,2,4,6],
       "ToyCar":[1,2,3,4],
       "ToyConveyor":[1,2,3]}
    
    for machineID in IDs[machinetype] :

        # Création du dataset d'entraînement
        train_data_byID = df[(df.Dataset == "train") & (df.Machine_Type == machinetype) & (df.Machine_ID == machineID)]
        train_data_stream_byID = dataset_stream(train_data_byID)

        # Création du dataset de test
        test_data_byID = df[(df.Dataset == 'test') & (df.Machine_Type == machinetype) & (df.Machine_ID == machineID)]
        test_data_stream_byID = dataset_stream(test_data_byID)

        # Création du vecteur label cible
        y_true_byID = test_data_byID['Status'].replace(['normal', 'anomaly'], [0,1])

        #Erreurs sur l'entraînement
        pred_train_data_byID = model_dense.predict(train_data_stream_byID)
        error_train_data_byID = errors(train_data_stream_byID, pred_train_data_byID, len(train_data_byID),nb_extract)
        errors_mean_byID = np.mean(error_train_data_byID)
        errors_std_byID = np.std(error_train_data_byID)

        #prédiction sur le dataset de test et évaluation des erreurs
        pred_test_data_byID = model_dense.predict(test_data_stream_byID)
        error_test_data_byID = errors(test_data_stream_byID, pred_test_data_byID, len(test_data_byID),nb_extract)

        #sauvegarde des erreurs
        np.save("TrainErrors_DenseAE_"+machinetype+'_'+str(machineID),error_train_data_byID)
        np.save("TestErrors_DenseAE_"+machinetype+'_'+str(machineID),error_test_data_byID)
    
        # calcul des 2 métriques d'évaluation du modèle
        auc = roc_auc_score(y_true_byID, error_test_data_byID)
        p_auc = roc_auc_score(y_true_byID, error_test_data_byID, max_fpr=0.1)
    
        # Calcul des F1-scores et du precentile maximisant la somme
        # Sauvegarde des F1-scores et de la somme
        percentile = f1scores(error_train_data_byID,
                                     error_test_data_byID,
                                     y_true_byID,
                                     machinetype,machineID)
    
        #seuil choisi à partir du percentile maximisant la somme des F1-scores
        seuil = np.percentile(error_train_data_byID,percentile)
    
        #création du vecteur des classes prédites
        y_pred = np.where(error_test_data_byID[:] > seuil, 1, 0)
    
        #création de la matrice de confusion
        conf_matrix = matrice_confusion(y_true_byID, y_pred)
    
        #calcul de la précision de détection d'anomalies
        precision = (matrice_confusion(y_true_byID, y_pred)[1][1]/y_pred.sum())
    
        #enregistrement des données d'évaluation
        list_to_save = []
        list_to_save.append(errors_mean_byID)
        list_to_save.append(errors_std_byID)
        list_to_save.append(auc)
        list_to_save.append(p_auc)
        list_to_save.append(percentile)
        list_to_save.append(seuil)
        list_to_save.append(conf_matrix[0][0])
        list_to_save.append(conf_matrix[0][1])
        list_to_save.append(conf_matrix[1][0])
        list_to_save.append(conf_matrix[1][1])
        list_to_save.append(precision)
        np.save("EvalData_DenseAE_"+machinetype+'_'+str(machineID),np.asarray(list_to_save))
    
    return "Fini !"