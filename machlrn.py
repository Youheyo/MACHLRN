import numpy as np
import pandas as pd

data = pd.read_csv('animal dataset.csv')

X = data.iloc[:, -1]
y = data.iloc[:, :-1]

def train_test_split(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def euclidean_distance(point1, point2):
	dist = (point1 - point2) ** 2
	return np.sqrt(dist)

def knn(X_train, y_train, X_Test, k):
	y_prediction = []
	for test_point in X_test.values:
		distances = []
		for train_point in X_train.values:
			distance = euclidean_distance(test_point, train_point)
			distances.append(distance)
		nearest_indices = sorted(range(len(list(distances))), key=lambda i: distances[i])[:k]
		nearest_labels = [y_train.iloc[i] for i in nearest_indices]
		predicted_label = max(set(nearest_labels), key=nearest_labels.count)
		y_prediction.append(predicted_label)
	return y_prediction

# Assuming K=3 for the KNN algorithm
k = 3
train_test_split(X, y)

y_pred = knn(X_train, y_train, X_test, k)

# Step 6: Calculate the accuracy of the model
def accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true.iloc[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)

accuracy = accuracy(y_test, y_pred)
print("Accuracy:", accuracy)