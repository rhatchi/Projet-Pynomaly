import librosa
import sys
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, BatchNormalization, ReLU, Reshape
import os
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler




#########################################
#
#             Preprocessing
#
#########################################



# Fonction de chargement d'un fichier audio
def load_audio(audio_path):
    return librosa.load(audio_path, sr=None)

def file_to_vector_array(filepath, sr=16000, n_mels=128, frames=5, n_fft=1024, hop_length=512, power=1.0):
    """
    transforme filepath en un vecteur array.
    filepath : chemin du fichier
    retourne : numpy.array( numpy.array(float))
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    y, _ = load_audio(filepath)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)

    # convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    vector_array = log_mel_spectrogram.T

    return vector_array

# définit un StandardScaler à appliquer ensuite sur le dataset 
def scaler(set_files):
    """
    set_files est un dataframe, donnt la première colonne contient les chemins
    ici, on va prendre set_files = train (d'une machinetype)
    """
    scaler = StandardScaler()
    liste = []
    for i in range(len(set_files)): 
        for vect in file_to_vector_array(set_files.iloc[i,0]):
            liste.append(vect)
    return scaler.fit(liste)


# transforme un chemin en un vecteur array normalisé
def file_to_vector_array_norm(filepath, scaler, sr=16000, n_mels=128, frames=5, n_fft=1024, hop_length=512, power=1.0):
    vector_array = file_to_vector_array(filepath, sr, n_mels, frames, n_fft, hop_length, power)
    vector_array = scaler.transform(vector_array)
    return vector_array


# préparer le jeu de données
def file_to_vector_ConvAE(filepath, scaler, sr=16000, n_mels=128, frames=5, n_fft=1024, hop_length=512, power=1.0):
    """
    retourne un vecteur de dimensions (105, 32, 128, 1) pour un fichier ToyCar et (95, 32, 128, 1) pour les autres
    """
    vector_array = file_to_vector_array_norm(filepath, scaler, sr, n_mels, frames, n_fft, hop_length, power)
    length, _ = vector_array.shape
    dim = 32
    step = 3
    idex = np.arange(length-dim+step, step=step)
    for idx in range(len(idex)):
        start = min(idex[idx], length - dim)
        vector = vector_array[start:start+dim,:]
        vector = vector.reshape((1, vector.shape[0], vector.shape[1]))
        if idx==0:
            batch = vector
        else:
            batch = np.concatenate((batch, vector))
                        
    # add channels dimension
    data = batch.reshape((batch.shape[0], batch.shape[1], batch.shape[2], 1))
    return data

# crée un dataset exploitable pour la suite
def dataset_conv(set_files, scaler, sr=16000, n_mels=128, frames=5, n_fft=1024, hop_length=512, power=1.0):
    """
    renvoie une dataset sur laquelle entrainer/évaluer le modèle Conv_AE
    set_files est un DataFrame et sa première colonne contient les chemins
    """
    # iterate file_to_vector_ConvAE()
    for k in range(len(set_files)):
        vector = file_to_vector_ConvAE(set_files.iloc[k,0], scaler, sr, n_mels, frames, n_fft, hop_length, power)
        if k == 0:
            X = np.empty((len(set_files)*vector.shape[0], vector.shape[1], vector.shape[2], vector.shape[3]))
            i = 0
            for l in vector:
                X[i,] = l
                i += 1
        else :
            i = k*vector.shape[0]
            for l in vector:
                X[i,] = l
                i += 1 
    return X


##########################
##### Recap function #####
##########################

def feature_selection(machinetype):
    # Chargement du dataframe des chemins d'accès
    df = pd.read_csv('Path_DF.csv')
    
    # Création du dataset d'entraînement
    train_data = df[(df.Dataset == "train") & (df.Machine_Type == machinetype)]
    scaler = scaler(train_data)
    train_data_conv = dataset_conv(train_data, scaler)
    
    # Création du dataset de test
    test_data = df[(df.Dataset == 'test') & (df.Machine_Type == machinetype)]
    test_data_conv = dataset_conv(test_data, scaler)
    
    # Création du vecteur label cible
    y_true = test_data['Status'].replace(['normal', 'anomaly'], [0,1])
    
    # Sauvegarde des data steams
    np.save("TrainPreprocessedData_ConvAE_"+machinetype+'_',train_data_conv)
    np.save("TestPreprocessedData_ConvAE_"+machinetype+'_',test_data_conv)
    
    return train_data, test_data, train_data_conv, test_data_conv, y_true








#########################################
#
#             Model
#
#########################################


def model_convAE(input_shape = (32, 128, 1)):
    model_conv = Sequential()

    # Première couche Encoder
    model_conv.add(Conv2D(filters = 32, kernel_size = (5, 5), strides = (1,2), padding = 'same', input_shape = input_shape))
    model_conv.add(BatchNormalization())
    model_conv.add(ReLU())

    # Seconde couche Encoder
    model_conv.add(Conv2D(filters = 64, kernel_size = (5, 5), strides = (1,2), padding = 'same'))
    model_conv.add(BatchNormalization())
    model_conv.add(ReLU())

    # Troisième couche Encoder
    model_conv.add(Conv2D(filters = 128, kernel_size = (5, 5), strides = (2,2), padding = 'same'))
    model_conv.add(BatchNormalization())
    model_conv.add(ReLU())

    # Quatrième couche Encoder
    model_conv.add(Conv2D(filters = 256, kernel_size = (3, 3), strides = (2,2), padding = 'same'))
    model_conv.add(BatchNormalization())
    model_conv.add(ReLU())

    # Cinquième couche Encoder
    model_conv.add(Conv2D(filters = 512, kernel_size = (3, 3), strides = (2,2), padding = 'same'))
    model_conv.add(BatchNormalization())
    model_conv.add(ReLU())

    # Couche goulot
    model_conv.add(Conv2D(filters = 40, kernel_size = (4, 4), strides = (1,1), padding = 'valid'))
    model_conv.add(Flatten())

    # Première couche Decoder
    model_conv.add(Dense(4*4*512))
    model_conv.add(Reshape((4,4,512)))

    # Seconde couche Decoder
    model_conv.add(Conv2DTranspose(filters = 256, kernel_size = (3, 3), strides = (2,2), padding = 'same'))
    model_conv.add(BatchNormalization())
    model_conv.add(ReLU())

    # Troisième couche Decoder
    model_conv.add(Conv2DTranspose(filters = 128, kernel_size = (3, 3), strides = (2,2), padding = 'same'))
    model_conv.add(BatchNormalization())
    model_conv.add(ReLU())

    # Quatrième couche Decoder
    model_conv.add(Conv2DTranspose(filters = 64, kernel_size = (5, 5), strides = (2,2), padding = 'same'))
    model_conv.add(BatchNormalization())
    model_conv.add(ReLU())

    # Cinquième couche Decoder
    model_conv.add(Conv2DTranspose(filters = 32, kernel_size = (5, 5), strides = (1,2), padding = 'same'))
    model_conv.add(BatchNormalization())
    model_conv.add(ReLU())

    # Couche de reconstruction 
    model_conv.add(Conv2DTranspose(filters = 1, kernel_size = (5, 5), strides = (1,2), padding = 'same'))
    
    return model_conv

    

##########################
##### Recap function #####
##########################

def model_conv_training(machinetype,train_data_conv):
    # Instanciation du modèle et compilation
    model_conv = model_convAE()
    model_conv.compile(loss = 'mean_squared_error', optimizer = 'adam')
    
    # Callback pour arrêter l'entrainement
    # et récupérer le meilleur modèle si la métrique ne diminue plus pendant 10 epochs
    early_stopping = callbacks.EarlyStopping(monitor = 'val_loss',
                                             patience = 10,
                                             mode = 'min',
                                             restore_best_weights = True)
    
    # Callback pour sauvegarder le meilleur modèle
    checkpoint = callbacks.ModelCheckpoint(filepath = os.getcwd()+'\\convAE_'+machinetype+'.hdf5',
                                           monitor = 'val_loss', 
                                           save_best_only = True,
                                           mode = 'min')
    
    # Entraînement du modèle
    history = model_conv.fit(train_data_conv, train_data_conv,
                batch_size = 64,
                epochs = 3,
                callbacks=[checkpoint, early_stopping],
                validation_split = 0.3)
    # Sauvegarde des fonctions de perte au cours de l'entraînement
    np.save("HistoryValLoss_ConvAE_"+machinetype,np.asarray(history.history['val_loss']))
    np.save("HistoryLoss_ConvAE_"+machinetype,np.asarray(history.history['loss']))
    
    return model_conv









#########################################
#
#             Evaluation
#
#########################################
    
# Fonction de calcul des erreurs entre le jeu de départ et les prédictions du modèle
def errors(X_true, X_pred, length, nb_extract = 85): 
    """
    WARNING : nb_extract = 105 pour ToyCar
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

def model_evaluation(machinetype,model_conv,
                     train_data,test_data,
                     train_data_conv,test_data_conv,
                     train_data_len,test_data_len,
                     y_true, nb_extract = 85):
        
    if machinetype == "ToyCar" : nb_extract = 105
    #Erreurs sur l'entraînement
    pred_train_data = model_conv.predict(train_data_conv)
    error_train_data = errors(train_data_conv, pred_train_data, train_data_len,nb_extract)
    errors_mean = np.mean(error_train_data)
    errors_std = np.std(error_train_data)
    
    #prédiction sur le dataset de test et évaluation des erreurs
    pred_test_data = model_conv.predict(test_data_conv)
    error_test_data = errors(test_data_conv, pred_test_data, test_data_len,nb_extract)
    
    for machineID in train_data.Machine_ID.unique() :
        
        #restriction des données à l'ID correspondant
        error_train_data_byID = error_train_data[train_data.Machine_ID == machineID]
        error_test_data_byID = error_test_data[test_data.Machine_ID == machineID]
        y_true_byID = y_true[test_data.Machine_ID == machineID]
        
        #sauvegarde des erreurs
        np.save("TrainErrors_ConvAE_"+machinetype+'_'+str(machineID),error_train_data_byID)
        np.save("TestErrors_ConvAE_"+machinetype+'_'+str(machineID),error_test_data_byID)
    
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
        list_to_save.append(errors_mean)
        list_to_save.append(errors_std)
        list_to_save.append(auc)
        list_to_save.append(p_auc)
        list_to_save.append(percentile)
        list_to_save.append(seuil)
        list_to_save.append(conf_matrix[0][0])
        list_to_save.append(conf_matrix[0][1])
        list_to_save.append(conf_matrix[1][0])
        list_to_save.append(conf_matrix[1][1])
        list_to_save.append(precision)
        np.save("EvalData_ConvAE_"+machinetype+'_'+str(machineID),np.asarray(list_to_save))
    
    return "Fini !"
