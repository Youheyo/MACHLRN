import numpy as np
import pandas as pd

def Split_Data(data, label, test_size=0.2):
    print("Splitting Data for", test_size * 100 , "% test size")
    split_index = int(len(data) * (1 - test_size))
    X_train, X_test = data[:split_index], data[split_index:]
    y_train, y_test = label[:split_index], label[split_index:]
    return X_train, X_test, y_train, y_test

# TODO: Create a split that randomizes the size

# * Old Way of loading dataset and contains original dataset
# ! Utilizes Original Dataset where the risk taker tag is undefined and has extra text
# ! Should still be usable with current code but haven't tested
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

# Split features and labels
featureset = dataset.iloc[:, 4:-1]
labelset = dataset.iloc[:, -1]

# * SPLITTING OF DATA
# Split the dataset for training and testing
X_train, X_test, y_train, y_test = Split_Data(featureset, labelset, test_size=0.8)
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

    def train(self, X, y, learning_rate=0.1, iter=100):

        for _ in range(iter):
            for i, _ in enumerate(X):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                
                # Update weights and bias based on prediction error
                self.weights += learning_rate * (y_true - y_pred) * x
                self.bias += learning_rate * (y_true - y_pred)
            
        print("TRAINING COMPLETE with..\n", learning_rate * 100, "% learning rate and", iter, "iterations")

# Create a perceptron
perceptron = Perceptron(num_features=X_train.shape[1])

# * Train the perceptron
perceptron.train(X_train, y_train, learning_rate=0.05,iter=100)

# Testing

def Accuracy_Test(X_test):
    accuracy = 0
    tp = fp = tn = fn = 0
    test_num = 1
    for i, _ in enumerate(X_test):
        x = X_test[i]
        y_true = y_test[i]
        y_pred = perceptron.predict(x)

        
        if y_pred == y_true == 1:
            tp += 1
            accuracy += 1
        elif y_pred == 1 and y_true == 0:
            fp += 1
        elif y_pred == 0 and y_true == 1:
            fn += 1
        elif y_pred == y_true == 0:
            tn += 1
            accuracy += 1
        test_num+=1

    accuracy /= len(X_test)
    
    print("---------------\nTEST COMPLETE")
    print("Accuracy:", accuracy)
    print("True Positive:", tp, "\tFalse Positive:", fp, "\nFalse Negative:", fn,"\tTrue Negative:", tn)

# ? Not sure if needed but it'd be nice especially if have to present
# TODO Refactor to allow user input

Accuracy_Test(X_test)
