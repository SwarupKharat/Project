import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('heart disease.csv')

heart_data.head()

heart_data.isnull().sum()

heart_data['target'].value_counts()

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


model = LogisticRegression()

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

input_data = ([[62,0,0,140,268,0,0,160,0,3.6,0,2,2]])


prediction = model.predict(input_data)

# if (prediction == 0):
#   print('The Person does not have a Heart Disease')
# else:
#   print('The Person has Heart Disease')

#pickle file of model
pickle.dump(model, open("heartdisease.pkl", "wb"))