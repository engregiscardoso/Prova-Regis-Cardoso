# -*- coding: utf-8 -*-
"""
Processo Seletivo 00518/2021 - Pesquisador II  

Inteligência Artificial

Instituto Senai de Inovação em Sistemas Embarcados

CANDIDATO: Regis Cardoso

"""
########################### BIBLIOTECAS ######################################
##############################################################################

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import fftpack
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime



######################## IMPORTAÇÃO NOVOS DATASETS  ##########################
##############################################################################

dfdata_ComTarget = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/dfinaltarget.csv')

dfdata_SemTarget = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/dffiltrado.csv')



########################## CLASSIFICAÇÃO COM SVM  ############################
##############################################################################

testeSVM = np.transpose(dfdata_ComTarget) 

trainSVM = np.transpose(dfdata_SemTarget) 

X_train, X_test, y_train, y_test = train_test_split(trainSVM, testeSVM.target, test_size=0.3,random_state=109) # 70% TREINO and 30% TESTE

# CLASSIFICADOR SVM #
clf = svm.SVC(kernel='linear') 

# TREINAMENTO DO MODELO #
clf.fit(X_train, y_train)

# PREDIÇOES #
y_pred = clf.predict(X_test)

# VERIFICAÇÃO DA ACURÁCIA
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test,y_pred))
print('\n')
print(confusion_matrix(y_test,y_pred))
