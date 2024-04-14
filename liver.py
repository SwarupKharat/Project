import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle

#load the data set using pandas
liver_dataset = pd.read_csv('liver disease.csv')

liver_dataset.head(10)
liver_dataset.tail(10)
liver_dataset.shape
liver_dataset.columns
liver_dataset['Gender'] = liver_dataset['Gender'].map({'Male': 1, 'Female': 2})
liver_dataset.head()
#we will check some statistical measures of the data
liver_dataset.describe()
liver_dataset.Dataset.value_counts()

list1=[]
for i in range(len(liver_dataset['Albumin_and_Globulin_Ratio'].values)):
    if pd.isnull(liver_dataset['Albumin_and_Globulin_Ratio'].values[i]) != True:
        list1.append(liver_dataset['Albumin_and_Globulin_Ratio'].values[i])
np.mean(list1)
liver_dataset['Albumin_and_Globulin_Ratio'].fillna(value=np.mean(list1),inplace=True)



X = liver_dataset.drop(columns='Dataset', axis=1)
Y = liver_dataset['Dataset']

X
Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)

X_train.shape
X_test.shape

X.shape


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model1 = LogisticRegression()
model1.fit(X_train, Y_train)


X_train_prediction = model1.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)



# accuracy on test data
X_test_prediction = model1.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(n_estimators = 100)
model2.fit(X_train,Y_train)

X_train_prediction2 = model2.predict(X_train)
training_data_accuracy2 = accuracy_score(X_train_prediction2, Y_train)



X_test_prediction2 = model2.predict(X_test)
test_data_accuracy2 = accuracy_score(X_test_prediction2, Y_test)


from sklearn.tree import DecisionTreeClassifier

model3 = DecisionTreeClassifier(random_state=101)
model3.fit(X_train, Y_train)

X_train_prediction3 = model3.predict(X_train)
training_data_accuracy3 = accuracy_score(X_train_prediction3, Y_train)



X_test_prediction3 = model3.predict(X_test)
test_data_accuracy3 = accuracy_score(X_test_prediction3, Y_test)



#Creating Data Frame for the accuracy score of logistic regression.

results_df = pd.DataFrame(data=[["Logistic Regression", training_data_accuracy*100, test_data_accuracy*100]],
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

#Creating Data Frame for the accuracy score of K-nearest neighbors.

results_df_2 = pd.DataFrame(data=[["Random Forest Classifier",training_data_accuracy2*100 , test_data_accuracy2*100]],
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])


#Creating Data Frame for the accuracy score of Decision Tree Classifier.

results_df_3 = pd.DataFrame(data=[["Decision Tree Classifier",training_data_accuracy3*100, test_data_accuracy3*100]],
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])


results_df

# Values from a Liver Scan
input_data = ([[62,2,1.7,140,268,0,0,160,0,3.6]])

# # change the input data to a numpy array
# input_data_as_numpy_array= np.asarray(input_data)
#
# # reshape the numpy array as we are predicting for only on instance
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#
# #predicting data using Random Forest Classifier
# prediction = model2.predict(input_data_reshaped)

# if (prediction[0]== 2):
#   print('According to the given details person does not have a Liver Disease')
# else:
#   print('According to the given details person has Liver Disease')

#pickle file model
pickle.dump(model2, open("liver.pkl", "wb"))
