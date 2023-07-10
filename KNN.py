import numpy as np
import pandas as pd

def Split_Data(data, label, test_size=0.2):
    print("Splitting Data for", test_size * 100 , "% test size")
    split_index = int(len(data) * (1 - test_size))
    X_train, X_test = data[:split_index], data[split_index:]
    y_train, y_test = label[:split_index], label[split_index:]
    return X_train, X_test, y_train, y_test

class KNN:
	def	__init__(self) -> None:
		pass

	def euclid_dist(self, point1, point2):
		dist = np.sqrt(np.sum((point1 - point2) ** 2))
		return dist
	
	def predict(self, X_train, y_train, X_test, k):
		y_pred = []
		for test_point in X_test:
			distances = []
			for train_point in X_train:
				distance =  self.euclid_dist(train_point, test_point)
				distances.append(distance)
			nearest_indices = sorted(range(len(distances)), key=lambda i:distances[i])[:k]
			nearest_labels = [y_train[i] for i in nearest_indices]
			predicted_label = max(set(nearest_labels), key=nearest_labels.count)
			y_pred.append(predicted_label)
		return y_pred

	def accuracy(self, y_true, y_pred):
		correct = 0
		for i in range(len(y_true)):
			if y_true[i] == y_pred[i]:
				correct += 1
		return correct / len(y_true)
	
# dataset = pd.read_csv('dataset/kNN Internet Survey Sheet_modified.csv')
# dataset = dataset.values.astype(float)
# np.random.shuffle(dataset)

# featureset = dataset[:, 4:-1] # ? Skips first 4 columns as per MC01 Specs
# labelset = dataset[:, -1] # ? Only takes last column aka labels

# X_train, X_test, y_train, y_test = Split_Data(featureset, labelset, test_size=0.2)

# knn = KNN()

# y_pred = knn.train(X_train, y_train, X_test, k=3)

# print(knn.accuracy(y_test, y_pred), "% Accuracy")
