# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 21:14:55 2019

@author: bronz0
"""
import pandas as pd
import pandas.plotting as plt2
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as hrch
import numpy as np
from sklearn import cluster
from sklearn import metrics
from sklearn.utils import shuffle

# convertir les labels en values numérique
def to_nums(labels):
    alphabet = ['AlifI', 'BaaI', 'TaaI', 'ThaI', 'JiimI', 'HaaI', 'KhaaI', 'DalI', 'DhelI', 'RaaI', 'ZadI', 'SiinI', 'ShiinI', 'SadI', 'DadI', 'ThaaI', 'DhaI', 'AiinI', 'GhiinI', 'FaaI', 'CaafI', 'KafI', 'LamI', 'MiimI', 'NounI', 'HaI', 'WawI', 'YaaI']
    nums = []
    for i in labels:
        nums.append(alphabet.index(i))
    return nums

# convertir les valeurs numérique en labels
def to_labels(nums):
    alphabet = ['AlifI', 'BaaI', 'TaaI', 'ThaI', 'JiimI', 'HaaI', 'KhaaI', 'DalI', 'DhelI', 'RaaI', 'ZadI', 'SiinI', 'ShiinI', 'SadI', 'DadI', 'ThaaI', 'DhaI', 'AiinI', 'GhiinI', 'FaaI', 'CaafI', 'KafI', 'LamI', 'MiimI', 'NounI', 'HaI', 'WawI', 'YaaI']
    labels = []
    for i in nums:
        labels.append(alphabet[i])
    return labels

# charegement de données
data = pd.read_table("C:\\Users\\pc\\Desktop\\master\\s2\\data mining\\tp\\tp2\\File_Features.txt", sep=",", header=0, index_col=31)
# dimension des données
print(data.shape)
#   Description statistique
data.describe

# randomize(shuffle) the dataset
data = shuffle(data)

# diviser le dataset en deux ensemble (80% entrainement, 20% test)
# et convertir les labels en valeurs numériques
y = data.index
x = data.values
x_train = np.array(x[:4480])
y_train = np.array(to_nums(y[:4480]))
x_test = np.array(x[4480:])
y_test = np.array(to_nums(y[4480:]))

#appliquer le Kmeans sur l'ensemble d'entrainement avec nombre de cluster = 28
kmeans = cluster.KMeans(n_clusters=28)
kmeans.fit(X=x_train, y=y_train)

#index triés des groupes
idk = np.argsort(kmeans.labels_)

#affichage des observations et leurs groupes
print(pd.DataFrame(data.index[idk],kmeans.labels_[idk]))

#distances aux centres de classes des observations
print(kmeans.transform(data))

#utilisation de la métrique "silhouette"
#faire varier le nombre de clusters de 2 à 28
res = []
wccs = []
for k in np.arange (10,27):
    km = cluster.KMeans(n_clusters =k)
    km.fit(x_train) 
    res.append(metrics.silhouette_score(x_train,km.labels_))
    wccs.append(kmeans.inertia_)
print(res)

# afficher les resultats
plt.title("Silhouette Score")
plt.xlabel("nombre de clusters")
plt.plot(np.arange(2,28,1), res)
plt.show()

# utiliser le model pour prédit les valeurs de l'ensemble de test 
y_pred = kmeans.fit_predict(x_test)

# convertir les valeurs numeriques en labels
y_test = np.array(to_labels(y_test))
y_pred = np.array(to_labels(y_pred))

# afficher le rapport de classification 
print(classification_report(y_test, y_pred))