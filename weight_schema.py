# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:17:25 2019

@author: User
"""

import numpy as np
import pandas as pd 

from sklearn.model_selection  import cross_val_score

from nltk.corpus import stopwords

import re
from sklearn.model_selection import train_test_split  

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from sklearn.model_selection import cross_validate 
import matplotlib.pyplot as plt
import seaborn as sns

neg_path = r"C:\Users\User\Desktop\머신러닝 팀플\dataset\rt-polaritydata\rt-polarity.neg.txt"
pos_path = r"C:\Users\User\Desktop\머신러닝 팀플\dataset\rt-polaritydata\rt-polarity.pos.txt"

def sentiment_raw_data(neg_path, pos_path):
    
    document = []
    
    with open(neg_path, "r") as f:
        for neg_sentence in f:
            document.append((neg_sentence, 0))
    
    with open(pos_path, "r") as f:
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

    complete_datset = [(processed_Data_[i], y[i]) for i in range(len(processed_Data_))]
        
    return complete_datset
    
def schema_comparion(neg_path, pos_path):
    
    overall_data =  sentiment_raw_data(neg_path, pos_path)
    
    text_data = [overall_data[i][0] for i in range(len(overall_data))]
    label = [overall_data[i][1] for i in range(len(overall_data))]
    
    X_train, X_test, y_train, y_test = train_test_split(text_data, label, test_size=0.2)
    
    ###########
    # Boolean #
    ###########
    
    # pipeline에 weighting scheme와 model 선언 #
    # CRF대신 RigdeClassifier 모델을 사용
    text_clf_SGDClassifier = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),lowercase=True),), 
                                       ('clf', SGDClassifier(loss='log'))],)
    
    text_clf_LogisticRegression = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),lowercase=True)),
                     ('clf', LogisticRegression(solver='lbfgs', multi_class='auto')),])
    
    text_clf_MultinomialNB = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),lowercase=True),),
                     ('clf', MultinomialNB()),])

    text_clf_RidgeClassifier = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), lowercase=True)),
                     ('clf', CalibratedClassifierCV(RidgeClassifier(solver='sparse_cg'), cv=5)),])
    
    # 어떤 metric을 가져올건지 선언 #
    scoring = ['precision_weighted','recall_weighted','f1_weighted','accuracy']
    
    SGD_Score = cross_validate(text_clf_SGDClassifier, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    LR_Score = cross_validate(text_clf_LogisticRegression, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    MNB_Score = cross_validate(text_clf_MultinomialNB, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    RC_Score = cross_validate(text_clf_RidgeClassifier, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    
    # model Metric #
    Boolean_recall_raw = SGD_Score['test_recall_weighted'].mean() + LR_Score['test_recall_weighted'].mean() \
                         + MNB_Score['test_recall_weighted'].mean() + RC_Score['test_recall_weighted'].mean()
    
    Boolean_precision_raw = SGD_Score['test_precision_weighted'].mean() + LR_Score['test_precision_weighted'].mean() \
                         + MNB_Score['test_precision_weighted'].mean() + RC_Score['test_precision_weighted'].mean()
    
    Boolean_f1_raw = SGD_Score['test_f1_weighted'].mean() + LR_Score['test_f1_weighted'].mean() \
                         + MNB_Score['test_f1_weighted'].mean() + RC_Score['test_f1_weighted'].mean()
   
    Boolean_accuracy_raw = SGD_Score['test_accuracy'].mean() + LR_Score['test_accuracy'].mean() \
                         + MNB_Score['test_accuracy'].mean() + RC_Score['test_accuracy'].mean()
    
    
    Boolean_recall = Boolean_recall_raw / len(scoring)
    Boolean_precision = Boolean_precision_raw / len(scoring)
    Boolean_f1 = Boolean_f1_raw / len(scoring)
    Boolean_accuracy = Boolean_accuracy_raw / len(scoring)
    
    ######
    # TF #
    ######
    
    TF_clf_SGDClassifier = Pipeline([
                     ('tfidf', TfidfVectorizer(use_idf=False,ngram_range=(1, 2))),\
                     ('clf', SGDClassifier(loss='log'))],)
    
    TF_clf_LogisticRegression = Pipeline([
                     ('tfidf', TfidfVectorizer(use_idf=False,ngram_range=(1, 2))),
                     ('clf', LogisticRegression(solver='lbfgs', multi_class='auto')),])
    
    TF_clf_MultinomialNB = Pipeline([
                     ('tfidf', TfidfVectorizer(use_idf=False,ngram_range=(1, 2))),
                     ('clf', MultinomialNB()),])
    
    TF_clf_RidgeClassifier = Pipeline([
                     ('tfidf', TfidfVectorizer(use_idf=False,ngram_range=(1, 2))),
                     ('clf', CalibratedClassifierCV(RidgeClassifier(solver='sparse_cg'), cv=5)),])
    
    SGD_Score_TF = cross_validate(TF_clf_SGDClassifier, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    LR_Score_TF = cross_validate(TF_clf_LogisticRegression, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    MNB_Score_TF = cross_validate(TF_clf_MultinomialNB, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    RC_Score_TF = cross_validate(TF_clf_RidgeClassifier, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    
    
    TF_recall_raw = SGD_Score_TF['test_recall_weighted'].mean() + LR_Score_TF['test_recall_weighted'].mean() \
                         + MNB_Score_TF['test_recall_weighted'].mean() + RC_Score_TF['test_recall_weighted'].mean()
    
    TF_precision_raw = SGD_Score_TF['test_precision_weighted'].mean() + LR_Score_TF['test_precision_weighted'].mean() \
                         + MNB_Score_TF['test_precision_weighted'].mean() + RC_Score_TF['test_precision_weighted'].mean()
    
    TF_f1_raw = SGD_Score_TF['test_f1_weighted'].mean() + LR_Score_TF['test_f1_weighted'].mean() \
                         + MNB_Score_TF['test_f1_weighted'].mean() + RC_Score_TF['test_f1_weighted'].mean()
    
    TF_accuracy_raw = SGD_Score_TF['test_accuracy'].mean() + LR_Score_TF['test_accuracy'].mean() \
                         + MNB_Score_TF['test_accuracy'].mean() + RC_Score_TF['test_accuracy'].mean()
    
   
    TF_recall = TF_recall_raw / len(scoring)
    TF_precision = TF_precision_raw / len(scoring)
    TF_f1 = TF_f1_raw / len(scoring)
    TF_accuracy = TF_accuracy_raw / len(scoring)
    
    ##########
    # TF-IDF #
    ##########
    
    TF_IDF_clf_SGDClassifier = Pipeline([
                     ('tfidf', TfidfVectorizer(use_idf=True,ngram_range=(1, 2))),\
                     ('clf', SGDClassifier(loss='log'))],)
    
    TF_IDF_clf_LogisticRegression = Pipeline([
                     ('tfidf', TfidfVectorizer(use_idf=True,ngram_range=(1, 2))),
                     ('clf', LogisticRegression(solver='lbfgs', multi_class='auto')),])
    
    TF_IDF_clf_MultinomialNB = Pipeline([
                     ('tfidf', TfidfVectorizer(use_idf=True,ngram_range=(1, 2))),
                     ('clf', MultinomialNB()),])
    
    TF_IDF_clf_RidgeClassifier = Pipeline([
                     ('tfidf', TfidfVectorizer(use_idf=True,ngram_range=(1, 2))),
                     ('clf', CalibratedClassifierCV(RidgeClassifier(solver='sparse_cg'), cv=5)),])
    
    SGD_Score_TI = cross_validate(TF_IDF_clf_SGDClassifier, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    LR_Score_TI = cross_validate(TF_IDF_clf_LogisticRegression, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    MNB_Score_TI = cross_validate(TF_IDF_clf_MultinomialNB, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    RC_Score_TI = cross_validate(TF_IDF_clf_RidgeClassifier, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    
    TI_recall_raw = SGD_Score_TI['test_recall_weighted'].mean() + LR_Score_TI['test_recall_weighted'].mean() \
                         + MNB_Score_TI['test_recall_weighted'].mean() + RC_Score_TI['test_recall_weighted'].mean()
    
    TI_precision_raw = SGD_Score_TI['test_precision_weighted'].mean() + LR_Score_TI['test_precision_weighted'].mean() \
                         + MNB_Score_TI['test_precision_weighted'].mean() + RC_Score_TI['test_precision_weighted'].mean()
    
    TI_f1_raw = SGD_Score['test_f1_weighted'].mean() + LR_Score_TI['test_f1_weighted'].mean() \
                         + MNB_Score_TI['test_f1_weighted'].mean() + RC_Score_TI['test_f1_weighted'].mean()
    
    TI_accuracy_raw = SGD_Score_TI['test_accuracy'].mean() + LR_Score_TI['test_accuracy'].mean() \
                         + MNB_Score_TI['test_accuracy'].mean() + RC_Score_TI['test_accuracy'].mean()
    
    
    TF_IDF_recall = TI_recall_raw / len(scoring)
    TF_IDF_precision = TI_precision_raw / len(scoring)
    TF_IDF_f1 = TI_f1_raw / len(scoring)
    TF_IDF_accuracy = TI_accuracy_raw / len(scoring)
    
    ## 데이터 시각화 ##
    data = {"weighting_schema" : ["0/1","0/1","0/1","0/1","TF","TF","TF","TF","TF-IDF","TF-IDF","TF-IDF","TF-IDF"], \
            "numeric" : [Boolean_recall,Boolean_precision,Boolean_f1,Boolean_accuracy,
                         TF_recall,TF_precision,TF_f1,TF_accuracy, 
                         TF_IDF_recall, TF_IDF_precision, TF_IDF_f1, TF_IDF_accuracy],
            "metric" : ['Recall', 'Precsion', 'F1', 'Accuracy','Recall', 'Precsion', 'F1', 'Accuracy','Recall', 'Precsion', 'F1', 'Accuracy']}
    
    df = pd.DataFrame(data, columns=["weighting_schema","numeric",'metric'])
    
    sns.barplot(data=df, x="weighting_schema", y="numeric", hue='metric') # default : dodge=True
    plt.ylim(0.74, 0.78)
    plt.yticks([0.74, 0.745, 0.75, 0.755, 0.76, 0.765, 0.77, 0.775, 0.78])
    plt.show()
    
    return 0
    
    
schema_comparion(neg_path, pos_path)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    