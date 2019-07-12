# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 21:52:21 2019

@author: bronz0
"""
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import collections

# methodes
# convertir les labels en valeur numérique
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

# le chemin de repertoir
DATA_DIR = 'C:/Users/pc/Desktop/master/s2/data mining/tp/tp3/FinalDataSet/'

# les noms des images
images_names = [DATA_DIR+i for i in os.listdir(DATA_DIR)]

# tableau des images
images = []
for i in range(len(images_names)):
    # ouvrir l'image, la converti en noir et blanc et la redimenssionée (128x128) --> (28x28)
    images.append(Image.open(images_names[i]).convert('L').resize((28, 28), Image.ANTIALIAS))


# les etiquetes
labels = [i.split('_')[0] for i in os.listdir(DATA_DIR)]

# compter le nombre d'instance pour chaque caractere
items = [item for item, count in collections.Counter(labels).items() if count > 1]
counts = {}
for item in items:
    counts[item] = labels.count(item)


# randomiser la distribution des données
images, labels = shuffle(images, labels, random_state=0)

# convertir les images en matrices
data = []
for i in range(len(images)):
    data.append(np.array(images[i]))

# deviser les données en deux ensemble entrainement et test (80% entrainement 20% test)
# convertir les labels en valeurs numériques
x_train = np.array(data[:4480])
y_train = np.array(to_nums(labels[:4480]))
x_test = np.array(data[4480:])
y_test = np.array(to_nums(labels[4480:]))


#plt.imshow(x_train[0],cmap=plt.cm.binary)

# normalisation et redimenssion (matrice de 28x28 --> vecteur de 784)
x_train = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1)
x_test = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)

# le model
model = tf.keras.models.Sequential()
# input layer
#model.add(tf.keras.layers.Flatten()) # pour rendre les matrices plat (un vecteur de 2500 au lieu d'une matrice 50x50)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape= x_train.shape[1:]))
# hidden layers
for i in range (1):
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# output layer
model.add(tf.keras.layers.Dense(28, activation=tf.nn.softmax))

# etrainement de model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200)

val_loss, val_acc = model.evaluate(x_train, y_train)
print(val_loss)
print(val_acc)

model.save('epic_num_reader.model')

model = tf.keras.models.load_model('epic_num_reader.model')

# get predicted labels
predicted_labels = []
for i in range(predictions.shape[0]):
    predicted_labels.append(np.argmax(predictions[i]))
predicted_labels = to_labels(predicted_labels)
# real labels
real_labels = to_labels(y_test)
# methode pour ajuster les espaces lors de l'affichage 
def spaces(nb):
    str = ''
    for i in range(0,nb):
        str = str+' '
    return str
# comparaison
n = 0
for i in range(len(predicted_labels)):
    print(predicted_labels[i], spaces(8-len(predicted_labels[i])),'| ', real_labels[i], spaces(8-len(real_labels[i])),'| ', predicted_labels[i]==real_labels[i])
    if(predicted_labels[i]==real_labels[i]):
        n = n+1
print('pourcentage de predictions justes : ', n/len(predicted_labels))