# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:24:50 2018

@author: zakis
"""


import pickle

infile = open("fonction_std_scaling.pyc",'rb')
std_scale = pickle.load(infile)
infile.close()

infile = open("fonction_prediction.pyc",'rb')
voting_classifier = pickle.load(infile)
infile.close()