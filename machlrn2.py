import numpy as np
import pandas as pd

data = pd.read_csv('dataset/animal dataset.csv')\

name = data.iloc[:, 1] #Only get the name row
data = data.iloc[1:, 3:] #Skip First Row and first 3 columns

X = data.iloc[:, :-1] #Get all except last column
y = data.iloc[:, -1] #Get last column only

def train_test_split(X, y, test_size=0.2):
	split_index = int(len(X) * (1 - test_size))
	X_train, X_test = X[1:split_index], X[split_index:]
	y_train, y_test = y[1:split_index], y[split_index:]
	return X_train, X_test, y_train, y_test, split_index

X_train, X_test, y_train, y_test, split_index = train_test_split(X, y, 0.5)

# print("X Train list", X_train)
# print("X Test list", X_test)
# print("Y Train list", y_train)
# print("Y Test list", y_test)

def euclid_dist(point1, point2):
	dist = np.sqrt(np.sum((point1 - point2) ** 2))
	return dist

def knn(X_train, y_train, X_Test, k):
	testCount = 0
	y_pred = []
	for test_point in pd.DataFrame(X_Test).values:
		if(testCount > 0): print("- - - - NEXT PET - - - -")
		distances = []
		for train_point in X_train.values:
			distance = euclid_dist(test_point, train_point)
			distances.append(distance)
		nearest_indices = sorted(range(len(distances)), key=lambda i:distances[i])[:k]
		print("Nearest Indice of",name[testCount+split_index-1],":",nearest_indices)

		nearest_labels = [y_train.iloc[i] for i in nearest_indices]
		for i in range(k):
			print(name[nearest_indices[i]], "is", nearest_labels[i])

		#print("Nearest label of current pet",nearest_labels)

		predicted_label = max(set(nearest_labels), key=nearest_labels.count)
		y_pred.append(predicted_label)

		petTag = ""
		if(y_pred[testCount] == 0): petTag = "not a pet"
		else: petTag = "is a pet"
		print(name[testCount+split_index-1], "is predicted to be", petTag)
		testCount += 1

	print("Number of cases checked:", testCount)
	return y_pred

y_pred = (knn(X_train, y_train, X_test, 5))
print(y_pred)

def accuracy(y_true, y_pred):
	correct = 0
	for i in range(len(y_true)):
		if y_true.iloc[i] == y_pred[i]:
			correct += 1
	return correct / len(y_true)

accuracy = accuracy(y_test, y_pred)
print("Accuracy: ", accuracy)