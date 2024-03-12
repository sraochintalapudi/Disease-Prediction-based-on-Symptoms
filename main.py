# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
DATA_PATH = "dataset/Training.csv"
#removing null values
data = pd.read_csv(DATA_PATH).dropna(axis = 1)
# Encoding the target value into numerical
# value using LabelEncoder
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
print(data['prognosis'].head())
X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.2, random_state = 24)

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_train)
print(f"Accuracy using Support Vector Machine Algorithm is: {accuracy_score(y_train,svm_preds)*100}")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_preds=nb_model.predict(X_train)
print(f"Accuracy using Gaussian Naive Bayes Algorithm is: {accuracy_score(y_train,nb_preds)*100}")

rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
rf_preds=rf_model.predict(X_test)
#print(f"Accuracy using Random Forest Algorithm is: {accuracy_score(y_train,rf_preds)*100}")

cf_matrix = confusion_matrix(y_test, rf_preds)
plt.figure(figsize=(12, 8))

sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Random Forest")
plt.show()