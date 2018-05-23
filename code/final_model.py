# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:24:23 2018

@author: zakis
"""

from datas import std_scale, voting_classifier
from preprocessing_dataset import preprocess_dataset

import pandas as pd

#predict the user's class 
#returns a dataframe with the class of each customer id
def predict_user_class(X_test):
    X_test_preprocessed = preprocess_dataset(X_test)
    customerID = X_test_preprocessed.index
    X_test_preprocessed = std_scale.transform(X_test_preprocessed)
    return pd.DataFrame(voting_classifier.predict(X_test_preprocessed), index=customerID)
    
#Example to predict a user's class
df = pd.read_excel("Online Retail.xlsx")
X_test = df.sample(frac=0.2)
pred = predict_user_class(X_test)