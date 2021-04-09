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


########################### ANALISE EXPLORATÓTIA  ############################
##############################################################################

# IMPORTAÇÃO DO DATASET, DIVIDINDO EM 50 PARTES#

chunk_size=10
batch_no=1
for chunk in pd.read_csv(file,chunksize=chunk_size):
    chunk.to_csv('chunk'+str(batch_no)+'.csv',index=False)
    batch_no+=1


     
# CARREGAR DATASET POR PARTES #
            
df1 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk1.csv')
df2 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk2.csv')
df3 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk3.csv')
df4 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk4.csv')
df5 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk5.csv')
df6 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk6.csv')
df7 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk7.csv')
df8 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk8.csv')
df9 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk9.csv')
df10 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk10.csv')
df11 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk11.csv')
df12 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk12.csv')
df13 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk13.csv')
df14 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk14.csv')
df15 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk15.csv')
df16 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk16.csv')
df17 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk17.csv')
df18 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk18.csv')
df19 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk19.csv')
df20 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk20.csv')
df21 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk21.csv')
df22 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk22.csv')
df23 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk23.csv')
df24 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk24.csv')
df25 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk25.csv')
df26 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk26.csv')
df27 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk27.csv')
df28 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk28.csv')
df29 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk29.csv')
df30 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk30.csv')
df31 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk31.csv')
df32 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk32.csv')
df33 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk33.csv')
df34 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk34.csv')
df35 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk35.csv')
df36 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk36.csv')
df37 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk37.csv')
df38 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk38.csv')
df39 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk39.csv')
df41 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk41.csv')
df42 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk42.csv')
df43 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk43.csv')
df44 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk44.csv')
df45 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk45.csv')
df46 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk46.csv')
df47 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk47.csv')
df48 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk48.csv')
df49 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk49.csv')
df50 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk50.csv')



# MATRIZ TRANSPOSTA, POIS MUITAS COLUNAS GERAM UM CARREGAMENTO DE MEMÓRIA #

df1T = np.transpose(df1)
df2T = np.transpose(df2)
df3T = np.transpose(df3)
df4T = np.transpose(df4)
df5T = np.transpose(df5)
df6T = np.transpose(df6)
df7T = np.transpose(df7)
df8T = np.transpose(df8)
df9T = np.transpose(df9)
df10T = np.transpose(df10)
df11T = np.transpose(df11)
df12T = np.transpose(df12)
df13T = np.transpose(df13)
df14T = np.transpose(df14)
df15T = np.transpose(df15)
df16T = np.transpose(df16)
df17T = np.transpose(df17)
df18T = np.transpose(df18)
df19T = np.transpose(df19)
df20T = np.transpose(df20)
df21T = np.transpose(df21)
df22T = np.transpose(df22)
df23T = np.transpose(df23)
df24T = np.transpose(df24)
df25T = np.transpose(df25)
df26T = np.transpose(df26)
df27T = np.transpose(df27)
df28T = np.transpose(df28)
df29T = np.transpose(df29)
df30T = np.transpose(df30)
df31T = np.transpose(df31)
df32T = np.transpose(df32)
df33T = np.transpose(df33)
df34T = np.transpose(df34)
df35T = np.transpose(df35)
df36T = np.transpose(df36)
df37T = np.transpose(df37)
df38T = np.transpose(df38)
df39T = np.transpose(df39)
df40T = np.transpose(df40)
df41T = np.transpose(df41)
df42T = np.transpose(df42)
df43T = np.transpose(df43)
df44T = np.transpose(df44)
df45T = np.transpose(df45)
df46T = np.transpose(df46)
df47T = np.transpose(df47)
df48T = np.transpose(df48)
df49T = np.transpose(df49)
df50T = np.transpose(df50)


# SALVANDO O DATASET EM NOVAS PARTES, TROCANDO LINHAS POR COLUNAS PARA PERMITIR A MANIPULAÇÃO#

dt_total = pd.concat([df1OK, df2OK, df3OK, df4OK, df5OK], axis=1)
dt_total.columns = (['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', 
                     '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50'])

dt_total.to_csv('df1a5.csv',index=False)


dt_total2 = pd.concat([df6OK, df7OK, df8OK, df9OK, df10OK, df11OK, df12OK, df13OK], axis=1)
dt_total2.columns = (['51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', 
                     '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117'
                     , '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130','131'])
    
dt_total2.to_csv('df6a13.csv',index=False)

dt_total3 = pd.concat([df14OK, df15OK, df16OK, df17OK, df18OK, df19OK, df20OK, df21OK, df22OK, df23OK, df24OK], axis=1)
dt_total3.columns = (['132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', 
                      '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', 
                      '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', 
                      '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', 
                      '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221','222', '223', '224', '225', '226', '227', '228', '229', '230', '231',
                      '232', '233', '234', '235', '236', '237', '238', '239', '240', '241'])


dt_total3.to_csv('df14a24.csv',index=False)    

dt_total5 = pd.concat([df25OK, df26OK, df27OK, df28OK, df29OK, df30OK, df31OK, df32OK, df33OK, df34OK, df35OK, df36OK, df37OK, df38OK, df39OK, df40OK, df41OK, df42OK, df43OK, df44OK, df45OK, df46OK, 
                       df47OK, df48OK, df49OK, df50OK], axis=1)
    
dt_total5.columns = ([242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 
                      273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302,
                      303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 
                      333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 
                      363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 
                      393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 
                      423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 
                      453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 
                      483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501])

           
dt_total5.to_csv('df25a50.csv',index=False)


# COMO FOI UTILIZADO A MATRIZ TRANSPOSTA E TROCADO LINHAS POR COLUNAS, AGORA É POSSIVEL CARREGAR TODO DATASET#

df1_5 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/df1a5.csv')
df6_13 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/df6a13.csv')
df14_24 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/df14a24.csv')
df25_50 = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/df25a50.csv')

df_geral = pd.concat([df1_5, df6_13,df14_24, df25_50], axis=1)



# ACHAR SINAIS DEFEITUOSOS #
    
with open('database.csv') as csv_file:

    csv_reader = csv.DictReader(csv_file)

    csv_reader.__next__()
    
    for row in csv_reader:
        if (row["target"]) == '1':
            print(row["signal_id"])
            

# APÓS ANALISE EXPLORATÓRIA, FOI VERIFICADO UM DESBALANCEAMENTO DOS DADOS (COM E SEM DESCARGAS), DESSA FORMA FOI CRIADO UM NOVO DATASET#

df1_filtrado = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk1.csv')
df2_filtrado = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk21.csv')
df3_filtrado = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk23.csv')
df4_filtrado = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk28.csv')
df5_filtrado = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk29.csv')
df6_filtrado = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk30.csv')
df7_filtrado = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk44.csv')
df8_filtrado = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/chunk46.csv')

df1_filtradoT = np.transpose(df1_filtrado)
df2_filtradoT = np.transpose(df2_filtrado)
df3_filtradoT = np.transpose(df3_filtrado)
df4_filtradoT = np.transpose(df4_filtrado)
df5_filtradoT = np.transpose(df5_filtrado)
df6_filtradoT = np.transpose(df6_filtrado)
df7_filtradoT = np.transpose(df7_filtrado)
df8_filtradoT = np.transpose(df8_filtrado)

dt_tota_final_check = pd.concat([df1_filtradoT, df2_filtradoT, df3_filtradoT, df4_filtradoT, df5_filtradoT, df6_filtradoT, df7_filtradoT, df8_filtradoT], axis=1)
dt_tota_final_check.columns = (['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', 
                     '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79'])


# VERIFICAÇÃO GRAFICA DOS DADOS #

linha1 = df1_filtrado.iloc[2:3,0:800000].values
plt.figure(figsize=(10,5))
plt.hist(linha1, 5, rwidth=0.9)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(linha1)
plt.show()


#VERIFICAÇÃO DAS COMPONENTES DA SÉRIE#
            
dfdata = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/dfproblemaOK.csv')

dataparse = lambda dates: datetime.strptime(dates, '%Y-%m-%d')
base = pd.read_csv('/content/drive/MyDrive/Prova_Pratica_RegisCardoso/dfproblemaOK.csv', parse_dates=['data'], index_col = 'data', date_parser = dataparse)

results = seasonal_decompose(base['5'])
plt.xlabel('Tempo')
plt.title('Série completa')
results.observed.plot(figsize=(15,3))

plt.xlabel('Tempo')
plt.title('Residuo')
results.resid.plot(figsize=(15,3))

plt.xlabel('Tempo')
plt.title('Tendência')
results.trend.plot(figsize=(15,3))