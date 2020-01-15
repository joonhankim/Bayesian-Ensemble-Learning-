# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:50:08 2019

@author: User
"""
#import package
import numpy as np
from sklearn.model_selection  import cross_val_score
from nltk.corpus import stopwords

import re
import pandas as pd
from sklearn.model_selection import train_test_split  

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from time import time
import tarfile
import matplotlib.pyplot as plt
import seaborn as sns

neg_path = r"C:\Users\user\Desktop\머신러닝 팀플\dataset\rt-polaritydata\rt-polarity.neg.txt"
pos_path = r"C:\Users\user\Desktop\머신러닝 팀플\dataset\rt-polaritydata\rt-polarity.pos.txt"

#sentence allocation & text preprocessing
def sentiment_raw_data(n_path, p_path):
    document = []
    
    with open(n_path, "rb") as f:
        for neg_sentence in f:
            document.append((neg_sentence, 0))
    
    with open(p_path, "rb") as f:
        for pos_sentence in f:
            document.append((pos_sentence, 1))
    
    document = pd.DataFrame(document)

    X = document.iloc[:,0].values
    y = document.iloc[:,1].values

    processed_Data_ = []

    for Data in range(0, len(X)):  
        # Remove all the special characters
        processed_Data = re.sub(r'\W', ' ', str(X[Data]))
     
        # remove all single characters
        processed_Data = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_Data)
     
        # Remove single characters from the start
        processed_Data = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_Data) 
     
        # Substituting multiple spaces with single space
        processed_Data= re.sub(r'\s+', ' ', processed_Data, flags=re.I)
     
        # Removing prefixed 'b'
        processed_Data = re.sub(r'^b\ss+', '', processed_Data)
        
        # Converting to Lowercase
        processed_Data = processed_Data.lower()
     
        processed_Data_.append(processed_Data)

    
    # 전처리후 x,y통합
    complete_datset = [(processed_Data_[i], y[i]) for i in range(len(processed_Data_))]
        
    return complete_datset


# accuracy 측정용
def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    
    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
        
    t0 = time()
    
    sentiment_fit = pipeline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_test)
    train_test_time = time() - t0
    accuracy = accuracy_score(y_test, y_pred)
    
    print ("null accuracy: {0:.2f}%".format(null_accuracy*100))
    print ("accuracy score: {0:.2f}%".format(accuracy*100))
    
    if accuracy > null_accuracy:
        print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100))
    elif accuracy == null_accuracy:
        print ("model has the same accuracy with the null accuracy")
    else:
        print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
        
    print ("train and test time: {0:.2f}s".format(train_test_time))
    print ("-"*80)
    
    return accuracy, train_test_time
    
def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, (str,bytes)): 
            #스트링형이 아닌 이터러블의 경우 - 재귀한 후, extend
            result.extend(flatten(el)) #재귀 호출
        else:
            #이터러블이 아니거나 스트링인 경우 - 그냥 append
            result.append(el)
    return result


# Calculating Precision, Recall & F-measure

     
    
    
overall_data =  sentiment_raw_data(neg_path, pos_path)

text_data = [overall_data[i][0] for i in range(len(overall_data))]
label = [overall_data[i][1] for i in range(len(overall_data))]

X_train, X_test, y_train, y_test = train_test_split(text_data, label, test_size=0.2, random_state=0)

###########
# Bagging #
###########

base1 = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=0)
base2 = MultinomialNB(random_state=0)
base3 = CalibratedClassifierCV(RidgeClassifier(solver='sparse_cg'), cv=5, random_state=0)
base4 = SGDClassifier(loss='log',random_state=0)
#base5 = RandomForestClassifier(n_estimators=10, max_depth=2,random_state=0)

# 10회 수행, 9개의 bags
bagging_accuracy = []

bg_clf1=BaggingClassifier(base_estimator=base1, n_estimators=9)
bg_clf2=BaggingClassifier(base_estimator=base2, n_estimators=9)
bg_clf3=BaggingClassifier(base_estimator=base3, n_estimators=9)
bg_clf4=BaggingClassifier(base_estimator=base4, n_estimators=9)
#bg_clf5=BaggingClassifier(base_estimator=base5, n_estimators=9)

for clf, label in zip([bg_clf1, bg_clf2, bg_clf3, bg_clf4], ['Logistic Regression', 'Multinomial NB', 'RidgeClassifier', 'SGDClassifier']):
    
    checker_pipeline = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 2),lowercase=True)),
            ('classifier', clf)
        ])

    print ("Validation result for {}".format(label))
    print (clf)
    
    clf_accuracy,tt_time = accuracy_summary(checker_pipeline, X_train, y_train, X_test, y_test)
    bagging_accuracy.append(clf_accuracy)
    
#################
# Simple Voting #
#################
SV_accuracy = []

clf1 = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=0)
clf2 = MultinomialNB(random_state=0)
clf3 = CalibratedClassifierCV(RidgeClassifier(solver='sparse_cg'), cv=5, random_state=0)
clf4 = SGDClassifier(loss='log', random_state=0)
#clf5 = RandomForestClassifier(n_estimators=10, max_depth=2,random_state=0)

eclf = VotingClassifier(estimators=[('lr', clf1), ('mnb', clf2), ('rc', clf3), ('SGD', clf4)], voting='hard')

for clf, label in zip([eclf], ['Simple Voting']):
    
    checker_pipeline = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 2),lowercase=True)),
            ('classifier', clf)
        ])

    print ("Validation result for {}".format(label))
    print (clf)
    
    clf_accuracy,tt_time = accuracy_summary(checker_pipeline, X_train, y_train, X_test, y_test)
    SV_accuracy.append(clf_accuracy)
    
#######
# BMA #
#######
text_clf_SGDClassifier = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), lowercase=True)),
                 ('clf', SGDClassifier(loss='log', random_state=0)),])

text_clf_LogisticRegression = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),lowercase=True)),
                 ('clf', LogisticRegression(solver='lbfgs', multi_class='auto',  random_state=0)),])

text_clf_MultinomialNB = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),lowercase=True)),
                 ('clf', MultinomialNB()),])   

text_clf_RandomForestClassifier = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),lowercase=True)),
         ('clf', RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0)),])   
 
text_clf_RidgeClassifier = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),lowercase=True)),
                ('clf', CalibratedClassifierCV(RidgeClassifier(solver='sparse_cg', random_state=0), cv=5),)])

scoring = ['f1_weighted']

text_clf_SGDClassifier.fit(X_train, y_train)
text_clf_LogisticRegression.fit(X_train, y_train)
text_clf_MultinomialNB.fit(X_train, y_train)
text_clf_RidgeClassifier.fit(X_train, y_train)
text_clf_RandomForestClassifier.fit(X_train, y_train)

proba_SGD = text_clf_SGDClassifier.predict_proba(X_test)
proba_LR = text_clf_LogisticRegression.predict_proba(X_test)
proba_MNB = text_clf_MultinomialNB.predict_proba(X_test)
proba_RC = text_clf_RidgeClassifier.predict_proba(X_test)
proba_RF = text_clf_RandomForestClassifier.predict_proba(X_test)

SGD_Score = cross_validate(text_clf_SGDClassifier, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
LR_Score = cross_validate(text_clf_LogisticRegression, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
MNB_Score = cross_validate(text_clf_MultinomialNB, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
RC_Score = cross_validate(text_clf_RidgeClassifier, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
RF_Score = cross_validate(text_clf_RidgeClassifier, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)

SGD_f1 = SGD_Score['test_f1_weighted'].mean()
LB_f1 = LR_Score['test_f1_weighted'].mean()
MNB_f1 = MNB_Score['test_f1_weighted'].mean()
RC_f1 = RC_Score['test_f1_weighted'].mean()
RF_f1 = RF_Score['test_f1_weighted'].mean()

new_proba_SGD = proba_SGD * SGD_f1
new_proba_LR = proba_LR * LB_f1
new_proba_MNB = proba_MNB * MNB_f1
new_proba_RC = proba_RC * RC_f1
new_proba_RF = proba_RF * RF_f1

new_proba_sum = new_proba_SGD + new_proba_LR + new_proba_MNB + new_proba_RC

new_y_pred = []

for i in range(len(new_proba_sum)):
    
    if new_proba_sum[i][0] < new_proba_sum[i][1]:
        new_y_pred.append(1)

    else:
        new_y_pred.append(0)

BMA_accuracy = accuracy_score(y_test, new_y_pred)

list1 = [bagging_accuracy,SV_accuracy,BMA_accuracy]
   
numeric = flatten(list1)

data = {"Ensemble_accuracy" : ["Bagging(LR)","Bagging(MNB)","Bagging(RC)","Bagging(SGD)",
                                "Simple_Voting", "BMA"],

        "numeric" : numeric}

df = pd.DataFrame(data=data, columns=["Ensemble_accuracy","numeric"])

sns.barplot(data=df, x="Ensemble_accuracy", y="numeric") 

fig = plt.gcf() #현재 figure에 불러오기
fig.set_size_inches(12, 8) #크기 바꾸기(inch 단위)

#ax3.set_xlabel('BMA' , fontsize=30)
#ax3.set_ylabel(' ')

#plt.title('Accuracy',fontsize=30) 
plt.ylim(0.5, 0.6, 0.7, 0.8)
plt.yticks([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
plt.show()


# compute contribution of each classifier

def bunja_1(i_clf,j_clf):
    y_pred_i = i_clf.predict(X_test)
    y_pred_j= j_clf.predict(X_test)
    accuracy_i = (y_test == y_pred_i)
    accuracy_j =  (y_test ==y_pred_j)
    temp= np.concatenate((accuracy_i.reshape(-1, 1), accuracy_j.reshape(-1, 1)), axis= 1)
    true_true= temp[temp[:, 1]== True]
    true_true= (true_true.sum(axis= 1)== 2).sum()/ true_true.shape[0]
    true_false= temp[temp[:, 1]== False]
    true_false= (true_false.sum(axis= 1)== 1).sum()/ true_false.shape[0]
    bunja= true_true+ true_false
    return bunja

def bunmo_1(i_clf,j_clf):
    y_pred_i = i_clf.predict(X_test)
    y_pred_j= j_clf.predict(X_test)
    accuracy_i = (y_test == y_pred_i)
    accuracy_j =  (y_test == y_pred_j)
    temp= np.concatenate((accuracy_i.reshape(-1, 1), accuracy_j.reshape(-1, 1)), axis= 1)
    false_true= temp[temp[:, 1]== True]
    false_true= (false_true.sum(axis= 1)== 1).sum()/ false_true.shape[0]
    false_false= temp[temp[:, 1]== False]
    false_false= (false_false.sum(axis= 1)== 0).sum()/ false_false.shape[0]
    bunmo = false_true+ false_false
    return bunmo




def RIS(i_clf,cl_list):
    new_list = [i for i in cl_list if i != i_clf ]      
    bunja_list=[]
    bunmo_list=[]
    for i in range(len(new_list)):
        bunja=bunja_1(i_clf,new_list[i])
        bunja_list.append(bunja)
        bunmo=bunmo_1(i_clf,new_list[i])
        bunmo_list.append(bunmo)
    result_bunja=sum(bunja_list)
    result_bunmo=sum(bunmo_list)
    result_ris=result_bunja/result_bunmo
    return result_ris

#find best ensemble S
#backward elimination by ris

clf_list=[text_clf_SGDClassifier,text_clf_LogisticRegression,text_clf_MultinomialNB,text_clf_RandomForestClassifier,text_clf_RidgeClassifier]

ACC_list=[]   

# first step
RIS_list=[]
for i in clf_list:
    RIS_list.append(RIS(i,clf_list))
ACC= np.mean(RIS_list)
ACC_list.append(ACC)

for i in clf_list:
    if RIS(i,clf_list) == np.min(RIS_list):
        clf_list.remove(i)

np.max(ACC_list)

# second step
RIS_list_1=[]        
for i in clf_list:
    RIS_list_1.append(RIS(i,clf_list))
ACC= np.mean(RIS_list_1)
ACC_list.append(ACC)

for i in clf_list:
    if RIS(i,clf_list) == np.min(RIS_list_1):
        clf_list.remove(i)
np.max(ACC_list)

# third step
RIS_list_2=[]        
for i in clf_list:
    RIS_list_2.append(RIS(i,clf_list))
ACC= np.mean(RIS_list_2)
ACC_list.append(ACC)

for i in clf_list:
    if RIS(i,clf_list) == np.min(RIS_list_2):
        clf_list.remove(i)
np.max(ACC_list)

# fourth step
RIS_list_3=[]        
for i in clf_list:
    RIS_list_3.append(RIS(i,clf_list))
ACC= np.mean(RIS_list_3)
ACC_list.append(ACC)

#ppt 표 참고 (backward elimination)



