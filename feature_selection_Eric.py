import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pylab
import random
import string
import operator
import re
import copy
import sys
import csv
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.feature_selection import RFE, f_regression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

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



 '''average the RandomForest importance rank several times'''
importance=np.zeros(len(train[0]))

rf_num=3
for num in range(rf_num):
    X_feature=np.array(train)
    Y_classifier=np.array(classifier)
    clf = RandomForestClassifier(n_estimators=50, max_features='auto',n_jobs=-1, min_samples_split=5)
    #clf = RandomForestRegressor(n_estimators=100, max_features='auto',n_jobs=-1, min_samples_split=5)
    clf.fit(X_feature, Y_classifier)
    importance_temp = clf.feature_importances_
    importance=map(operator.add, importance, importance_temp)


importance=map(lambda x: x/rf_num,importance)




selected_feature_index=sorted(range(len(importance)), key=lambda i: importance[i])[-30:]
#selected_feature_index=[ 52, 40, 56, 558, 559] 

def reduced_data(train,selected_feature_index):
    '''give the feature_reduced training set'''
    train_reduced=[]
    for ele in train:
        temp=[ele[i] for i in selected_feature_index]
        train_reduced.append(temp)
    return train_reduced

train_reduced=reduced_data(train,selected_feature_index)
test_reduced=reduced_data(test,selected_feature_index)    

print selected_feature_index  
feature_score=map(lambda i: importance[i], selected_feature_index)
#print sorted(feature_score, reverse=True)
    


def Dev_accuracy(train_reduced,classifier):
    '''testing for accuracy using dev_set'''
    X_feature=np.array(train_reduced)
    Y_classifier=np.array(classifier)
#     regression = RandomForestRegressor(n_estimators=100,oob_score=True)
#     regression.fit(X_feature, Y_classifier)
#     out_of_bag_accuracy = regression.oob_score_
    clf = RandomForestClassifier(n_estimators=50,oob_score=True)
    clf.fit(X_feature, Y_classifier)
    out_of_bag_accuracy = clf.oob_score_
    
    return out_of_bag_accuracy

print Dev_accuracy(train_reduced,classifier)
print selected_feature_index
#print('accuracy:', clf.score(test_reduced, classifier_test))





''' recursively reduce the featrue'''
selected_feature=selected_feature_index[::-1]

def accuracy(train,classifier,selected_feature_index):
    train_reduced=reduced_data(train,selected_feature_index)
    accuracy=Dev_accuracy(train_reduced,classifier)
    return accuracy

final=[]
final_accuracy=[]
while len(final) < 5:
    pre_accuracy=0
    for ele in selected_feature:
        now_accuracy=accuracy(train,classifier,final+[ele])
        if now_accuracy > pre_accuracy:
            print ele, final, now_accuracy
            temp_index=ele
            pre_accuracy=now_accuracy
            
    final.append(temp_index)  
    final_accuracy.append(pre_accuracy)
    selected_feature.remove(temp_index)
    



 print final,final_accuracy,selected_feature
 



 '''Random forest calssification'''
#test1=[ 52, 202, 559, 516, 69]   # from others
#test1=[53, 508, 560, 52, 347]    # selected with 50 random forest tree
# train_reduced=reduced_data(train,test1)
# test_reduced=reduced_data(test,test1)

train_reduced=reduced_data(train,final)
test_reduced=reduced_data(test,final)


#clf = RandomForestRegressor(n_estimators=100,oob_score=True)
clf=RandomForestClassifier(n_estimators=100, oob_score=False)
clf.fit(train_reduced, classifier)


print('accuracy:', clf.score(test_reduced, classifier_test))

# '''Random forest'''
# clf = RandomForestClassifier(n_estimators=5)
# clf.fit(train_reduced, classifier)
# print('accuracy:', clf.score(test_reduced, classifier_test))




'''test of arbitrary accuray'''
try1 = [53, 54, 214, 231, 52]
train_reduced=reduced_data(train,try1)
test_reduced=reduced_data(test,try1)
clf = RandomForestClassifier(n_estimators=50)
#clf = RandomForestRegressor(n_estimators=20)

clf.fit(train_reduced, classifier)
print('accuracy:', clf.score(test_reduced,classifier_test))


       