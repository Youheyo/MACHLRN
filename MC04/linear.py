
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

print("Linear Regression")

try:
    pick_dataset = int(input("Pick Dataset [0,1]"))
    dataset_test_size = int(input("Enter Test size %"))
except:
    print("Error in inputs, Default parameters set")
    pick_dataset = 0
    dataset_test_size = .2

if(dataset_test_size > 1): dataset_test_size /= 100

print(f"Using dataset{pick_dataset+1} with a test size of {dataset_test_size*100}%")

#region Data setup

match pick_dataset:
    case 0:
        dataset = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
        diabetes_tag = dataset['Diabetes_binary']
    case 1:
        dataset = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
        diabetes_tag = dataset['Diabetes_012']

features = dataset.iloc[:, 1:]

scaler = preprocessing.StandardScaler()
scaler.fit(features)

features_scaled = scaler.transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, diabetes_tag, test_size=dataset_test_size, shuffle=True)

#endregion

data = LogisticRegression().fit(X_train,y_train)
y_pred = data.predict(X_test)

# score = data.score(X_test, y_test)
# print("Linear Regression Accuracy", score)

print(classification_report(y_test, y_pred,zero_division=np.nan))
matrix = confusion_matrix(y_test, y_pred,)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=data.classes_)
disp.plot(colorbar=False)
plt.show()

# tn, fp, fn, tp = matrix.ravel()
# print(f"True Positives: {tp}\tFalse Negatives: {fn}\nFalse Positives: {fp}\tTrue Negatives: {tn}")
print("Program Complete")
