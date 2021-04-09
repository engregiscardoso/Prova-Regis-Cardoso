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

dfdata_SemTarget = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/dffiltrado.csv')


######################## CLASSIFICAÇÃO COM K-MEANS  ##########################
##############################################################################

df = DataFrame(dfdata_SemTarget)  

# TREINAMENTO DO MODELO #
kmeans = KMeans (n_clusters = 2) .fit (df)

# CALCULO DOS CENTROIDES #
centroids = kmeans.cluster_centers_

# VALORES A SEREM AVALIADOS / COMPARADOS#
v1 = '29'
v2 = '64'

plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 16})
plt.scatter(df[v1], df[v2], c= kmeans.labels_.astype(float), s=30, alpha=3)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, label = 'teste')
plt.show()


# VERIFICAÇÃO ESTATISTICA#
estat1 = df[v1]
estat1.describe()

estat2 = df[v2]
estat2.describe()