import numpy as np
import pandas as pd

def Split_Data(data, label, test_size=0.2):
    split_index = int(len(data) * (1 - test_size))
    X_train, X_test = data[:split_index], data[split_index:]
    y_train, y_test = label[:split_index], label[split_index:]
    return X_train, X_test, y_train, y_test


"""
# Load the dataset
    dataset = pd.read_csv('dataset/kNN Internet Survey Sheet.csv')

    # Remove the column labels from the first row and the first four columns
    dataset = dataset.iloc[1:, 4:]

    # Split the dataset into training and testing sets
    training_data = dataset[103:].values.astype(float)
    testing_data = dataset[:100].values.astype(float)

    # Extract the features and labels from the training data
    X_train = training_data[:, :-3]
    y_train = training_data[:, -3]

    # Extract the features and labels from the testing data
    X_test = testing_data[:, :-3]
    y_test = testing_data[:, -4]
"""

dataset = pd.read_csv('dataset/kNN Internet Survey Sheet_modified.csv')

#dataset = dataset.iloc[1: :]
featureset = dataset.iloc[:, 4:-1]
labelset = dataset.iloc[:, -1]

X_train, X_test, y_train, y_test = Split_Data(featureset, labelset,0.2)
X_train, X_test, y_train, y_test = X_train.values.astype(float), X_test.values.astype(float), y_train.values.astype(float), y_test.values.astype(float)


class Perceptron:
    def __init__(self, num_features):
        # Initialize the weights and bias to zeros
        self.weights = np.zeros(num_features)
        self.bias = 0

    def activation(self, x):
        # Step function as the activation function
        return 1 if x >= 0 else 0

    def predict(self, x):
        # Compute the weighted sum of inputs and apply the activation function
        z = np.dot(self.weights, x) + self.bias
        return self.activation(z)

    def train(self, X, y, learning_rate=0.1, num_epochs=100):
        testNum = 0
        for _ in range(num_epochs):
            for i, _ in enumerate(X):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                # Update weights and bias based on prediction error
                self.weights += learning_rate * (y_true - y_pred) * x
                self.bias += learning_rate * (y_true - y_pred)
            testNum += 1
        print("Number of cases trained", testNum)

# Create a perceptron model
perceptron = Perceptron(num_features=X_train.shape[1])

# Train the model
perceptron.train(X_train, y_train)

# Test the model

def Accuracy_Test(X_test):
    accuracy = 0
    testNum = 1
    for i, _ in enumerate(X_test):
        x = X_test[i]
        y_true = y_test[i]
        y_pred = perceptron.predict(x)

        if y_pred == y_true:
            accuracy += 1
        testNum+=1

    accuracy /= len(X_test)
    print("Accuracy test ran for ", testNum)
    return accuracy

print("Accuracy:", Accuracy_Test(X_test))
