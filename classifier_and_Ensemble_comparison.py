# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:50:49 2019

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

#tar 풀기
#train_tar=tarfile.open(r"C:\Users\eric\Desktop\unprocessed.tar.gz", 'r')
#train_tar.extractall(r'C:\Users\eric\Desktop\text_data')

neg_path = r"C:\Users\user\Desktop\머신러닝 팀플\dataset\rt-polaritydata\rt-polarity.neg.txt"
pos_path = r"C:\Users\user\Desktop\머신러닝 팀플\dataset\rt-polaritydata\rt-polarity.pos.txt"

#doc_neg = r"C:\Users\eric\Desktop\text_data\sorted_data\music\negative.txt"
#doc_pos = r"C:\Users\eric\Desktop\text_data\sorted_data\music\positive.txt"


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

def classifier_Ensemble_comparison(n_path, p_path):
    
    overall_data =  sentiment_raw_data(neg_path, pos_path)
    
    text_data = [overall_data[i][0] for i in range(len(overall_data))]
    label = [overall_data[i][1] for i in range(len(overall_data))]
    
    X_train, X_test, y_train, y_test = train_test_split(text_data, label, test_size=0.2, random_state=0)
    
    
    clf11 = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),lowercase=True)),
                     ('clf', LogisticRegression(solver='lbfgs', multi_class='auto')),])
    
    clf22 = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),lowercase=True),),
                     ('clf', MultinomialNB()),])

    clf33 = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), lowercase=True)),
                     ('clf', CalibratedClassifierCV(RidgeClassifier(solver='sparse_cg'), cv=5)),])
    
    clf44 = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),lowercase=True),), 
                                       ('clf', SGDClassifier(loss='log'))],)
    
    clf55 = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), lowercase=True)),
                     ('clf', RandomForestClassifier(n_estimators=10, max_depth=2)),])
    
    # classifier # 
    
    clf1_LR = clf11.fit(X_train,y_train)
    clf2_MNB = clf22.fit(X_train,y_train)
    clf3_RC = clf33.fit(X_train,y_train)
    clf4_SGD = clf44.fit(X_train,y_train)
    clf5_RF = clf55.fit(X_train,y_train)
    
    LR_Accuracy= clf1_LR.score(X_test, y_test)
    MNB_Accuracy = clf2_MNB.score(X_test, y_test)
    RC_Accuracy = clf3_RC.score(X_test, y_test)
    SDG_Accuracy = clf4_SGD.score(X_test, y_test)
    RFC_Accuracy = clf5_RF.score(X_test, y_test)
    
    data1 = {"Classifer" : ["SGD","LR","MNB","RC","RFC"], \
            "numeric" : [SDG_Accuracy,LR_Accuracy,MNB_Accuracy,RC_Accuracy,RFC_Accuracy]}
    
    df1 = pd.DataFrame(data1, columns=["Classifer","numeric"])
    
    
    #############################
    
    # Simple Voting #
    # CRF -> RC, DIC -> RF, ME - > LR, NB -> MNB, SVM -> SGD  
    # 성능이 다른 모델에 비해 떨어지는 RF를 위주로 modeling # 
    
    clf1 = LogisticRegression(solver='lbfgs', multi_class='auto')
    clf2 = MultinomialNB()
    clf3 = CalibratedClassifierCV(RidgeClassifier(solver='sparse_cg'), cv=5)
    clf4 = SGDClassifier(loss='log')
    clf5 = RandomForestClassifier(n_estimators=10, max_depth=2,random_state=0)
    
    eclf1 = VotingClassifier(estimators=[('lr', clf1),('RF',clf5)], voting='hard')
    eclf2 = VotingClassifier(estimators=[('mnb', clf2),('RF',clf5)], voting='hard')
    eclf3 = VotingClassifier(estimators=[('rc', clf3),('RF',clf5)], voting='hard')
    eclf4 = VotingClassifier(estimators=[('SGD', clf4),('RF',clf5)], voting='hard')
    
    eclf5 = VotingClassifier(estimators=[('lr', clf1), ('mnb', clf2),('RF',clf5)], voting='hard')
    eclf6 = VotingClassifier(estimators=[('mnb', clf2), ('rc', clf3),('RF',clf5)], voting='hard')
    eclf7 = VotingClassifier(estimators=[('rc', clf3), ('SGD', clf4),('RF',clf5)], voting='hard')
    eclf8 = VotingClassifier(estimators=[('lr', clf1), ('SGD', clf4),('RF',clf5)], voting='hard')
    eclf9 = VotingClassifier(estimators=[('mnb', clf2),('SGD', clf4),('RF',clf5)], voting='hard')
    eclf10 = VotingClassifier(estimators=[('lr', clf1), ('rc', clf3), ('RF',clf5)], voting='hard')
    
    eclf11 = VotingClassifier(estimators=[('mnb', clf2), ('rc', clf3), ('SGD', clf4),('RF',clf5)], voting='hard')
    eclf12 = VotingClassifier(estimators=[('lr', clf1),  ('rc', clf3), ('SGD', clf4),('RF',clf5)], voting='hard')
    eclf13 = VotingClassifier(estimators=[('lr', clf1), ('mnb', clf2), ('SGD', clf4),('RF',clf5)], voting='hard')
    eclf14 = VotingClassifier(estimators=[('lr', clf1), ('mnb', clf2), ('rc', clf3) ,('RF',clf5)], voting='hard')
    
    eclf15 = VotingClassifier(estimators=[('lr', clf1), ('mnb', clf2), ('rc', clf3), ('SGD', clf4),('RF',clf5)], voting='hard')
    
    Voting_accuracy = []
    
    for clf, label in zip([eclf1,eclf2,eclf3,eclf4,eclf5,
                           eclf6,eclf7,eclf8,eclf9,eclf10,
                           eclf11,eclf12,eclf13,eclf14,eclf15], 
                    ["Simple Voting1","Simple Voting2","Simple Voting3","Simple Voting4",
                     "Simple Voting5","Simple Voting6","Simple Voting7","Simple Voting8",
                     "Simple Voting9","Simple Voting10","Simple Voting11","Simple Voting12",
                     "Simple Voting13","Simple Voting14","Simple Voting15"]):
        
        checker_pipeline = Pipeline([
                ('vect', CountVectorizer(ngram_range=(1, 2),lowercase=True)),
                ('classifier', clf)
            ])
    
        print ("Validation result for {}".format(label))
        print (clf)
        
        clf_accuracy, tt_time = accuracy_summary(checker_pipeline, X_train, y_train, X_test, y_test)
        Voting_accuracy.append(clf_accuracy)
    
    data2 = {"SV" : ["LR_RF","MNB_RF","RC_RF","SGD_RF","LR_MNB_RF",
                           "MNB_RC_RF","RC_SGD_RF","LR_SGD_RF","MNB_SGD_RF","LR_RC_RF",
                           "MNB_RC_SGD_RF","LR_RC_SGD_RF","LR_MNB_SGD_RF","LR_MNB_RC_RF","LR_MNB_RC_SGD_RF"], \
            "numeric" : Voting_accuracy}
    
    
    df2 = pd.DataFrame(data2, columns=["SV","numeric"])

    #############################
    
    # BMA #
    
    proba_LR = clf1_LR.predict_proba(X_test)
    proba_MNB = clf2_MNB.predict_proba(X_test)
    proba_RC = clf3_RC.predict_proba(X_test)
    proba_SGD = clf4_SGD.predict_proba(X_test)
    proba_RF = clf5_RF.predict_proba(X_test)    
    
    scoring = ['f1_weighted']
    
    SGD_Score = cross_validate(clf44, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    LR_Score = cross_validate(clf11, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    MNB_Score = cross_validate(clf22, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    RC_Score = cross_validate(clf33, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    RF_Score = cross_validate(clf55, X_train, y_train, scoring=scoring , cv=10, return_train_score=True)
    
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
    
    new_proba_sum_1 = new_proba_MNB + new_proba_RF
    new_proba_sum_2 = new_proba_RC + new_proba_RF
    new_proba_sum_3 = new_proba_SGD + new_proba_RF
    new_proba_sum_4 = new_proba_LR + new_proba_RF
    
    new_proba_sum_5 = new_proba_MNB + new_proba_LR + new_proba_RF
    new_proba_sum_6 = new_proba_MNB + new_proba_RC + new_proba_RF
    new_proba_sum_7 = new_proba_SGD + new_proba_RC + new_proba_RF
    new_proba_sum_8 = new_proba_SGD + new_proba_LR + new_proba_RF
    new_proba_sum_9 = new_proba_SGD + new_proba_MNB + new_proba_RF
    new_proba_sum_10 = new_proba_LR + new_proba_RC + new_proba_RF
    
    new_proba_sum_11 = new_proba_SGD + new_proba_MNB + new_proba_RC + new_proba_RF
    new_proba_sum_12 = new_proba_SGD + new_proba_LR + new_proba_RC + new_proba_RF
    new_proba_sum_13 = new_proba_SGD + new_proba_LR + new_proba_MNB + new_proba_RF
    new_proba_sum_14 = new_proba_LR + new_proba_MNB + new_proba_RC + new_proba_RF
    new_proba_sum_15 = new_proba_SGD + new_proba_LR + new_proba_MNB + new_proba_RC + new_proba_RF
    
    new_proba_sum_ = [new_proba_sum_1,new_proba_sum_2,new_proba_sum_3,new_proba_sum_4,new_proba_sum_5,
                      new_proba_sum_6,new_proba_sum_7,new_proba_sum_8,new_proba_sum_9,new_proba_sum_10,
                      new_proba_sum_11,new_proba_sum_12,new_proba_sum_13,new_proba_sum_14,new_proba_sum_15]
    
    all_bma_acc = []
   
    for j in range(len(new_proba_sum_)):
        
        new_y_pred = []
        
        for i in range(len(new_proba_sum_1)):
            
            if new_proba_sum_[j][i][0] < new_proba_sum_[j][i][1]:
                new_y_pred.append(1)
        
            else:
                new_y_pred.append(0)
            
        bma_acc = accuracy_score(y_test, new_y_pred)
        all_bma_acc.append(bma_acc)
    
    data3 = {"BMA" : ["LR_RF","MNB_RF","RC_RF","SGD_RF","LR_MNB_RF",
                           "MNB_RC_RF","RC_SGD_RF","LR_SGD_RF","MNB_SGD_RF","LR_RC_RF",
                           "MNB_RC_SGD_RF","LR_RC_SGD_RF","LR_MNB_SGD_RF","LR_MNB_RC_RF","LR_MNB_RC_SGD_RF"], \
            "numeric" : all_bma_acc}
    
    df3 = pd.DataFrame(data3, columns=["BMA","numeric"])
    
    # Classifier Plot #
    ax1 = sns.barplot(data=df1, x="Classifer", y="numeric",) # default : dodge=True
    ax1.set_xlabel('Classifier', fontsize=20)
    ax1.set_ylabel(' ')
    plt.title('Accuracy',fontsize=20)
    plt.ylim(0.5, 0.6, 0.7 , 0.8)
    plt.yticks([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    plt.show()
    
    # Simple Vooting Plot #
    ax2 = sns.barplot(data=df2, x="SV", y="numeric") 
    fig = plt.gcf() #현재 figure에 불러오기
    fig.set_size_inches(24, 10) #크기 바꾸기(inch 단위)
    ax2.set_xlabel('Simple Voting',fontsize=30)
    ax2.set_ylabel(' ')
    
    plt.title('Accuracy',fontsize=30) 
    plt.ylim(0.5, 0.6, 0.7, 0.8)
    plt.yticks([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    plt.show()
    
    # BMA Plot #
    ax3 = sns.barplot(data=df3, x="BMA", y="numeric")
    fig = plt.gcf() #현재 figure에 불러오기
    fig.set_size_inches(24, 10) #크기 바꾸기(inch 단위)
    
    ax3.set_xlabel('BMA' , fontsize=30)
    ax3.set_ylabel(' ')
    
    plt.title('Accuracy',fontsize=30) 
    plt.ylim(0.5, 0.6, 0.7, 0.8)
    plt.yticks([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    plt.show()
    
    return 0
    
    
classifier_Ensemble_comparison(neg_path, pos_path)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    