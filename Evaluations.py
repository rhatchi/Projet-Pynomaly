import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


##########################################################
# Fonction de perte au cours de l'entraînement du modèle #
##########################################################

def courbe_loss(history_loss, history_val_loss, machinetype, xlim=None, ylim=None):
    # Labels et limites des axes
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    # Limites des axes
    if xlim!=None : plt.xlim(xlim)
    if ylim!=None : plt.ylim(ylim)

    # Traçage de la courbe de la précision sur l'échantillon de validation
    plt.plot([x for x in range(len(history_loss))],
             history_loss, 
             label = 'Loss',
             color = 'red')    
    
    # Traçage de la courbe de la précision sur l'échantillon d'entrainement
    plt.plot([x for x in range(len(history_val_loss))],
             history_val_loss,
             label = 'Validation Loss',
             color = 'blue')
    
    #Titre
    plt.title("Évolution de la fonction de perte lors de l'apprentissage sur la machine "+machinetype)
    # Affichage de la légende
    plt.legend()
    
    # Affichage de la figure
    return plt.show()

##########################################################
# Calcul des données à partir des fichiers de sauvegarde #
##########################################################

def get_y_true(path_csv, machinetype, machineID):
    df = pd.read_csv(path_csv)
    return df[(df.Dataset == "test") & (df.Machine_Type == machinetype) & (df.Machine_ID == machineID)]['Status'].replace(['normal', 'anomaly'], [0,1])
  

def get_y_pred(percentile, errors_train, errors_test):
    seuil = np.percentile(errors_train,percentile)
    return np.where(errors_test[:] > seuil, 1, 0)


################################################################
# Fonctions d'affichage de graphiques sur les données d'une ID #
################################################################
    
# ROC Curve
def courbe_ROC(y_true, errors_test, machinetype, machineID):
    # calcul des données
    fpr, tpr, seuils = roc_curve(y_true, errors_test, pos_label = 1)
    roc_auc = auc(fpr, tpr)
    
    # Définition des limites des axes
    plt.xlim(0,1)
    plt.ylim(0,1.05)
    
    # Traçage des courbes
    plt.plot(fpr, tpr, 'orange', label = 'Modèle conv. (auc = %0.2f)' % roc_auc)
    plt.plot(fpr, fpr, 'b--', label = 'Aléatoire (aux = 0.5)')
    
    #Titre
    plt.title('Courbe ROC associée à la machine '+machinetype+'_'+machineID)
    
    #Noms des axes
    plt.xlabel('Taux faux positifs')
    plt.ylabel('Taux vrais positifs')
    
    # Affichage de la légende
    plt.legend(loc = 'lower right')
    
    # Affichage de la figure
    return plt.show()

    

# F1-scores
def courbes_f1score(f1_1, f1_0, somme, machinetype, machineID):
    # Traçage des courbes
    plt.plot(range(101),f1_1,label="Anomaly")
    plt.plot(range(101),f1_0,label="Normal")
    plt.plot(range(101),somme,label="Somme")
    
    # Noms des axes
    plt.xlabel('Percentile de seuil')
    plt.ylabel('F1-score')
    
    #Titre
    plt.title('Courbes des F1-scores associés à la machine '+machinetype+'_'+machineID)
    
    # Affichage de la légende
    plt.legend()
    
    # Affichage de la figure
    plt.show()
    
    return print('\nLe score F1 maximum de la classe \"Anomaly\" est',max(f1_1).round(3),
          'et est atteint lorsque le seuil est au percentile',np.where(f1_1==max(f1_1))[0][0],
          '\n\nLe score F1 maximum de la classe \"Normal\" est',max(f1_0).round(3),
          'et est atteint lorsque le seuil est au percentile',np.where(f1_0==max(f1_0))[0][0],
          '\n\nLa somme maximum des scores F1 est',max(somme).round(3),
          'et est atteinte lorsque le seuil est au percentile',np.where(somme==max(somme))[0][0])



########################
# Matrice de confusion #
########################

def matrice_confusion(y_true, y_pred):
    return pd.crosstab(y_true, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])

