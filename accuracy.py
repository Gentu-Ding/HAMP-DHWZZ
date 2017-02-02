'''Author: Li Hsin Cheng
   9 features selected with accuracy 0.86
   improvement on SVM needed '''
   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pylab
import random
import string
import re
import copy
import sys
import csv
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


df=open("/Users/lihsin/Desktop/Class/semester2/data practicum/Human Activity Recognition/UCI HAR Dataset/train/X_train.txt")
df1=open("/Users/lihsin/Desktop/Class/semester2/data practicum/Human Activity Recognition/UCI HAR Dataset/train/y_train.txt")

content = df.read().splitlines()
classify = df1.read().splitlines()

train=[]
classifier=[]
for ele in content:
    feature=ele.split(' ')
    temp=[]
    for i in range(0,len(feature)):
        if feature[i] != '' and feature[i] != ' ':
            temp.append((feature[i]))
    train.append(temp)        


    
for ele in classify:
    classifier.append(int(ele))



df3=open("/Users/lihsin/Desktop/Class/semester2/data practicum/Human Activity Recognition/UCI HAR Dataset/test/X_test.txt")
df4=open("/Users/lihsin/Desktop/Class/semester2/data practicum/Human Activity Recognition/UCI HAR Dataset/test/y_test.txt")

content_test = df3.read().splitlines()
classify_test = df4.read().splitlines()

test=[]
classifier_test=[]
for ele in content_test:
    feature=ele.split(' ')
    temp=[]
    for i in range(0,len(feature)):
        if feature[i] != '' and feature[i] != ' ':
            temp.append((feature[i]))
    test.append(temp)        


    
for ele in classify_test:
    classifier_test.append(int(ele))




X_feature=np.array(train)
Y_classifier=np.array(classifier)
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_feature, Y_classifier)
importance = clf.feature_importances_
print importance



selected_feature=sorted(range(len(importance)), key=lambda i: importance[i])[-9:]

'''only take the selected features'''
train_reduced=[]
for ele in train:
    temp=[ele[i] for i in selected_feature]
    train_reduced.append(temp)
    
test_reduced=[]
for ele in test:
    temp=[ele[i] for i in selected_feature]
    test_reduced.append(temp)
    
    
    '''SVM calssification'''
clf = SVC(probability=False, C=100)
clf.fit(train_reduced, classifier)


print('accuracy:', clf.score(test_reduced, classifier_test))

# '''Random forest'''
# clf = RandomForestClassifier(n_estimators=5)
# clf.fit(train_reduced, classifier)
# print('accuracy:', clf.score(test_reduced, classifier_test))





