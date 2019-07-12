# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:35:05 2019

@author: bronz0
"""
import pandas as pd
import numpy as np  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC  



# charegement de données
data = pd.read_csv("C:\\Users\\pc\\Desktop\\master\\s2\\data mining\\tp\\tp2\\File_Features.txt")

# division de l'ensemble de données
X = data.drop('label', axis=1)
y = data['label'] 
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# entrainement avec linear kernel
print("Linear Kernel ************************************************************")
model = SVC(kernel='linear')  
model.fit(X_train, y_train)  

# utiliser le model pour prédit les valeurs de l'ensemble de test
y_pred = model.predict(X_test)  

# afficher les resultats
#print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred)) 
print("Accuracy = ",accuracy_score(y_test, y_pred, normalize=True))  

# entrainement avec kernel polynomial
print("Polynomial Kernel ************************************************************")

model = SVC(kernel='poly', degree=5)  
model.fit(X_train, y_train)  

# utiliser le model pour prédit les valeurs de l'ensemble de test
y_pred = model.predict(X_test)  

#print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred)) 
print("Accuracy = ",accuracy_score(y_test, y_pred, normalize=True)) 

# entrainement avec kernel gaussian
print("Gaussian Kernel ************************************************************")
model = SVC(kernel='rbf')  
model.fit(X_train, y_train)  

# utiliser le model pour prédit les valeurs de l'ensemble de test
y_pred = model.predict(X_test)  

#print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print("Accuracy = ",accuracy_score(y_test, y_pred, normalize=True)) 

# entrainement avec kernel sigmoid
print("Sigmoid Kernel ************************************************************")
model = SVC(kernel='sigmoid')  
model.fit(X_train, y_train)  

# utiliser le model pour prédit les valeurs de l'ensemble de test
y_pred = model.predict(X_test)  

#print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred)) 
print("Accuracy = ",accuracy_score(y_test, y_pred, normalize=True)) 
 