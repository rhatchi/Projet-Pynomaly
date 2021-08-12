import pandas as pd
import numpy as np
import librosa
from python_speech_features import mfcc, logfbank, ssc


def load_audio(audio_path):
    return librosa.load(audio_path, sr=None)

def feature_extraction(audio_path,features_param):
    
    signal, samplerate = load_audio(audio_path)
    
    logmelspec = mfcc(signal=signal,
                      samplerate=samplerate,
                      winlen=features_param["winlen"],
                      winstep=features_param["winstep"],
                      numcep=features_param["numcep"],
                      nfilt=features_param["nfilt"],
                      nfft=features_param["nfft"],
                      preemph=features_param["preemph"],
                      ceplifter=features_param["ceplifter"],
                      appendEnergy=features_param["appendEnergy"],
                      winfunc=features_param["winfunc"]).reshape(-1)
    
    fbank_feat = logfbank(signal=signal,
                          samplerate=samplerate,
                          winlen=features_param["winlen"],
                          winstep=features_param["winstep"],
                          nfilt=features_param["nfilt"],
                          nfft=features_param["nfft"],
                          preemph=features_param["preemph"]).reshape(-1)
    
    ssc_feat = ssc(signal=signal,
                   samplerate=samplerate,
                   winlen=features_param["winlen"],
                   winstep=features_param["winstep"],
                   nfilt=features_param["nfilt"],
                   nfft=features_param["nfft"],
                   preemph=features_param["preemph"],
                   winfunc=features_param["winfunc"]).reshape(-1)
    
    return np.concatenate((signal,logmelspec,fbank_feat,ssc_feat))


def preprocessing(csv,features_param):
    df = pd.read_csv(csv)
    #Choix du type de machine
    Machine_Types=df.Machine_Type.unique()
    question = "Choisisir le type de machine parmi :\n"+str(Machine_Types)
    machine_type = input(question)
    
    #Choix du machine_ID
    machine_IDs = df[(df.Machine_Type == machine_type)].Machine_ID.unique()
    question2 = "Choisisir l'ID de la machine :"+str(machine_IDs)
    machine_ID = int(input(question2))
    
    #Restriction du DF à la machine et au dataset choisis
    df_work = df[(df.Machine_Type == machine_type) & (df.Machine_ID == machine_ID)]
    #encodage de la variable cible "Status"
    df_work = df_work.replace(['normal', 'anomaly'], [0,1])
    
    #dimension des features
    signal, samplerate = load_audio(df_work.iloc[0,0])
    dim_mfcc = mfcc(signal=signal,
                    samplerate=samplerate,
                    winlen=features_param["winlen"],
                    winstep=features_param["winstep"],
                    numcep=features_param["numcep"],
                    nfilt=features_param["nfilt"],
                    nfft=features_param["nfft"],
                    preemph=features_param["preemph"],
                    ceplifter=features_param["ceplifter"],
                    appendEnergy=features_param["appendEnergy"],
                    winfunc=features_param["winfunc"]).shape
    
    dim_fbank = logfbank(signal=signal,
                         samplerate=samplerate,
                         winlen=features_param["winlen"],
                         winstep=features_param["winstep"],
                         nfilt=features_param["nfilt"],
                         nfft=features_param["nfft"],
                         preemph=features_param["preemph"]).shape
    
    dim_ssc = ssc(signal=signal,
                  samplerate=samplerate,
                  winlen=features_param["winlen"],
                  winstep=features_param["winstep"],
                  nfilt=features_param["nfilt"],
                  nfft=features_param["nfft"],
                  preemph=features_param["preemph"],
                  winfunc=features_param["winfunc"]).shape
    
    #dictionnaire retourné, contenant des Numpy Arrays de données brutes et features
    output={}
    
    #création d'un dataset rassemblant toutes les données
    data = pd.DataFrame(df_work['Path'].apply(lambda chemin : feature_extraction(chemin,features_param)).tolist(),index=df_work.index)
    data = pd.concat([df_work[['Dataset']],data],axis = 1)
    
    #séparation entre dataset d'entraînement et de test
    data_train = data[data.Dataset == 'train'].drop(['Dataset'],1)
    data_test = data[data.Dataset == 'test'].drop(['Dataset'],1)
    
    #vecteurs cible
    output['y_train'] = df_work[df_work.Dataset == 'train']['Status'].to_numpy()
    output['y_test'] = df_work[df_work.Dataset == 'test']['Status'].to_numpy()
    
    #audio brut
    len1=len(signal)
    output['X_train'] = data_train.iloc[:,:len1].to_numpy()
    output['X_test'] = data_test.iloc[:,:len1].to_numpy()
  
    #features
    #mfcc
    len2=len1+dim_mfcc[0]*dim_mfcc[1]
    output['X_mfcc_train'] = data_train.iloc[:,len1:len2].to_numpy().reshape(-1,dim_mfcc[0],dim_mfcc[1])
    output['X_mfcc_test'] = data_test.iloc[:,len1:len2].to_numpy().reshape(-1,dim_mfcc[0],dim_mfcc[1])
    #fbank
    len3=len2+dim_fbank[0]*dim_fbank[1]
    output['X_fbank_train'] = data_train.iloc[:,len2:len3].to_numpy().reshape(-1,dim_fbank[0],dim_fbank[1])
    output['X_fbank_test'] = data_test.iloc[:,len2:len3].to_numpy().reshape(-1,dim_fbank[0],dim_fbank[1])
    #ssc
    output['X_ssc_train'] = data_train.iloc[:,len3:].to_numpy().reshape(-1,dim_ssc[0],dim_ssc[1])
    output['X_ssc_test'] = data_test.iloc[:,len3:].to_numpy().reshape(-1,dim_ssc[0],dim_ssc[1])
    
    return output
