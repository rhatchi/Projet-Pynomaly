import pandas as pd
import numpy as np
import librosa 
import sys

def file_load_stream(wav_name, mono=False):
    frameSize = librosa.get_samplerate(wav_name)
    hoplength = frameSize // 2
    stream = librosa.stream(wav_name, block_length=1, frame_length=frameSize, hop_length=hoplength, mono=mono)
    return stream


def file_to_vector_array_stream_test_data(file_name, n_mels=128, frames=5, n_fft=1024, hop_length=512, power=1):
    """
    convert file_name to a vector array.
    file_name : str
        target .wav file
    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames
    
    # 02 generate melspectrogram using librosa
    stream = file_load_stream(file_name)
    sr = librosa.get_samplerate(file_name)
    liste = []
    for n, y in enumerate(stream):
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)

        # 03 convert melspectrogram

        log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

        # 04 calculate total vector size
        vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1  
        
        # 05 skip too short clips
        if vector_array_size < 1:
            return np.empty((0, dims))

        # 06 generate feature vectors by concatenating multiframes
        vector_array = np.zeros((vector_array_size, dims))

        for t in range(frames):
            vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

        liste.append(vector_array)

    liste = np.asarray(liste)
    liste = liste.reshape(liste.shape[0] * liste.shape[1], liste.shape[2])
    return liste


def dataset_stream(set_files):
    """
    renvoie une dataset sur laquelle entrainer/évaluer le modèle Dense_AE
    set_files est un DataFrame et sa première colonne contient les chemins
    """
    liste = []
    for k in range(len(set_files)):
        for l in file_to_vector_array_stream_test_data(set_files.iloc[k,0]):
            liste.append(l)
    return np.asarray(liste)


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, BatchNormalization, ReLU, Reshape

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


def errors(X_true, X_pred, length, nb_extract = 532): 
    """
    calcule les erreurs entre le jeu de départ et les prédictions du modèle
    """
    vect_error = np.mean(np.square(X_true - X_pred), axis=1)
    errors = np.zeros(length)
    for k in range(length):
        errors[k] = np.mean(vect_error[k*nb_extract : (k+1)*nb_extract])
    return errors
