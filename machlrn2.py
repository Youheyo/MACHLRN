import numpy as np
import pandas as pd

data = pd.read_csv('animal dataset.csv')\

data = data.iloc[1:, 3:]

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# print(data)
# print(y)

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
		# print(predicted_label)
		y_pred.append(predicted_label)
	print(distances)
	print("Sorted Result")
	print(nearest_indices)
	return y_pred

sampleData = [1] * 20
# print(sampleData)

result = (knn(X, y, sampleData, 3))
# print(result)
if(result == 0):
	print("The animal is NOT a pet")
else:
	print("The animal is a pet")
