# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

### This file regroups non DEEP machine learning : we exclusively use scikit-learn package

##First step : divide the data array into a X array containing the numeric data and Y array containing only the label column ------

data = pd.read_pickle('./alldata.txt')

#Numeric data array
X = data.drop(['Label'], axis=1).as_matrix().astype(np.float)

np.isnan(X).any(), np.isinf(X).any() #verifying that there are not any invalid values.

#Label column
Y = data['Label']

#Let's check out the number of attack types contained in the Y column :
attackTypes = data['Label'].unique()
print(len(attackTypes))

#divide the global dataset into a training set and a test set : here we choose 70% for training and 30% for test

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


##Second step: training the data with the various method available---------------------------

#Logistic Regression

logisticRegressionClassifier = LogisticRegression().fit(X_train, y_train)
y_trainPredict = logisticRegressionClassifier.predict(X_train)
y_testPredict = logisticRegressionClassifier.predict(X_test)

#Decision Tree

decisionTreeClassifier = DecisionTreeClassifier().fit(X_train, y_train)
y_trainPredict_dt = decisionTreeClassifier.predict(X_train)
y_testPredict_dt = decisionTreeClassifier.predict(X_test)

#Random Forest

randomForestClassifier = RandomForestClassifier().fit(X_train, y_train)
y_trainPredict_rf = randomForestClassifier.predict(X_train)
y_testPredict_rf = randomForestClassifier.predict(X_test)

#Support Vector Machine : the calculation was extremely slow consequently we decided to leave this part commented.

#Support Vector Machine (very slow!!!)

#svmClassier = LinearSVC().fit(X_train, y_train)
#y_trainPredict_svm = svmClassier.predict(X_train)
#y_testPredict_svm = svmClassier.predict(X_test)


##Third step : save the trained models and the result arrays into pickles -----------------------------------------

