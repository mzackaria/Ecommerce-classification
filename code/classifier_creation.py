# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:16:03 2018

@author: zakis
"""



import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import preprocessing_dataset

df = pd.read_excel("Online Retail.xlsx")
y_kmeans = pd.read_csv('y_kmeans.csv')
customerID = y_kmeans.CustomerID
y_kmeans = y_kmeans.drop('CustomerID', 1)

customerID_train = customerID
df_train = df

X_train = df_train[np.in1d(df_train.CustomerID, customerID_train)]
X_train_preprocessed = preprocessing_dataset.preprocess_dataset(X_train)
customerID_train = X_train_preprocessed.index

sc_X = StandardScaler()
X_train_preprocessed = sc_X.fit_transform(X_train_preprocessed)

y_train = y_kmeans[np.in1d(customerID, customerID_train)].values

gbc = GradientBoostingClassifier()
rfc = RandomForestClassifier()
svc = SVC()

vc2 = VotingClassifier(estimators=[('rf', rfc), ('gbc', gbc), ('svc', svc)], voting='hard')
model = vc2.fit(X_train_preprocessed, y_train)

filehandler = open("fonction_std_scaling.pyc", 'wb')
pickle.dump(sc_X, filehandler)

filehandler = open("fonction_prediction.pyc", 'wb')
pickle.dump(model, filehandler)