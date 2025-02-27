import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.tree import DecisionTreeClassifier


pick_dataset = 0
dataset_test_size = .2
plot_tree = False

try:
    pick_dataset = int(input("Pick Dataset [0,1]: "))
    dataset_test_size = int(input("Enter Test size %: "))
    depth = int(input("Enter Depth: "))

except:
    print("Error in inputs, Default parameters set")
    pick_dataset = 1
    dataset_test_size = .2
    depth = 5

if(dataset_test_size > 1): dataset_test_size /= 100
if(depth == 0): depth = None

print(f"Using dataset{pick_dataset+1} with a test size of {dataset_test_size*100}%\nMax Depth of tree is {(depth)}")

#region Data setup

match pick_dataset:
    case 0:
        dataset = pd.read_csv("MC04/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
        diabetes_tag = dataset['Diabetes_binary']
    case 1:
        dataset = pd.read_csv("MC04/diabetes_012_health_indicators_BRFSS2015.csv")
        diabetes_tag = dataset['Diabetes_012']

features = dataset.iloc[:, 1:]

scaler = preprocessing.StandardScaler().fit(features)

features_scaled = scaler.transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, diabetes_tag, test_size=dataset_test_size,shuffle=True)

#endregion

data = DecisionTreeClassifier(max_depth=depth).fit(X_train, y_train)

if(plot_tree): tree.plot_tree(data)

y_pred = data.predict(X_test)

if(depth == None): print("Max depth by tree: ",data.get_depth())
# print("Decision Tree Accuracy : ", data.score(X_test, y_test ))
print(classification_report(y_test, y_pred,zero_division=np.nan))
matrix = confusion_matrix(y_test, y_pred, labels=data.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=data.classes_)

disp.plot(colorbar=False)
plt.show()
