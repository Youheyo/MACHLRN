import numpy as np
import pandas as pd

data = pd.read_csv('animal dataset.csv')\

data = data.iloc[1:, 3:]

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

def train_test_split(X, y, test_size=0.2):
	split_index = int(len(X) * (1 - test_size))
	X_train, X_test = X[1:split_index], X[split_index:]
	y_train, y_test = y[1:split_index], y[split_index:]
	return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)

# print("X Train list", X_train)
# print("X Test list", X_test)
# print("Y Train list", y_train)
# print("Y Test list", y_test)

def euclid_dist(point1, point2):
	dist = np.sqrt(np.sum((point1 - point2) ** 2))
	return dist

def knn(X_train, y_train, X_Test, k):
	y_pred = []
	for test_point in pd.DataFrame(X_Test).values:
		distances = []
		for train_point in X_train.values:
			distance = euclid_dist(test_point, train_point)
			distances.append(distance)
		nearest_indices = sorted(range(len(distances)), key=lambda i:distances[i])[:k]
		nearest_labels = [y_train.iloc[i] for i in nearest_indices]
		predicted_label = max(set(nearest_labels), key=nearest_labels.count)
		y_pred.append(predicted_label)
	print("Distances list ", distances)
	print("Nearest Indices" , nearest_indices)
	return y_pred

y_pred = (knn(X_train, y_train, X_test, 3))
print(y_pred)

def accuracy(y_true, y_pred):
	correct = 0
	for i in range(len(y_true)):
		if y_true.iloc[i] == y_pred[i]:
			correct += 1
	return correct / len(y_true)

accuracy = accuracy(y_test, y_pred)
print("Accuracy: ", accuracy)