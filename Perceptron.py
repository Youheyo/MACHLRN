#!/usr/bin/env python3

import numpy as np
import pandas as pd

def Split_Data(data, label, test_size=0.2):
    print("Splitting Data for", test_size * 100 , "% test size")
    split_index = int(len(data) * (1 - test_size))
    X_train, X_test = data[:split_index], data[split_index:]
    y_train, y_test = label[:split_index], label[split_index:]
    return X_train, X_test, y_train, y_test

# ! Current dataset has a random risk taker tag attached which was done via excel

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

def Accuracy_Test(X_test):
    accuracy = 0
    tp = fp = tn = fn = 0
    test_num = 1
    for i, _ in enumerate(X_test):
        x = X_test[i]
        y_true = y_test[i]
        y_pred = perceptron.predict(x)
    
        if y_pred == 1 and y_true == 1:
            tp += 1
            accuracy += 1
        elif y_pred == 1 and y_true == 0:
            fp += 1
        elif y_pred == 0 and y_true == 1:
            fn += 1
        elif y_pred == 0 and y_true == 0:
            tn += 1
            accuracy += 1
        test_num+=1

    accuracy /= len(X_test)
    
    print("---------------\nTEST COMPLETE")
    print("Total Cases:", tp+fp+fn+tn)
    print("True Positive:", tp, "\tFalse Negative:", fn, "\nFalse Positive:", fp,"\tTrue Negative:", tn)
    
    print("Precision:", tp/(tp+fp))
    print("Accuracy:", accuracy)
    print("Recall:", tp/(tp+fn))
    print("Specificity:", tn/(tn+fp))
    print("F-Measure:", (2 * (tp/(tp+fp)) * (tp/(tp+fn)) / ((tp/(tp+fp)) + (tp/(tp+fn))) ) )

#Loading Dataset
dataset = pd.read_csv('dataset/MCO1 New InternetSurveyDataset.csv')
dataset = dataset.values.astype(float)

#User Input
try:
    user_input_test_size = int(input("Enter Test size (20 - 80): "))
    if user_input_test_size < 20 or user_input_test_size > 80:
        print("Entered Test size was incompatible. Default values are set")
        user_input_test_size = 0.2
    else:
        user_input_test_size /= 100
except (TypeError, ValueError):
    print("Input was invalid. Default values are set")
    user_input_test_size = 0.2

try:
    user_learning_rate = int(input("Enter Learning Rate:"))
    if(user_learning_rate <= 0):
        print("Learning Rate invalid. Default Values are set")
        user_learning_rate = 0.01
    else:
        user_learning_rate /= 100
except (TypeError, ValueError):
    print("Input was invalid. Default values are set")
    user_learning_rate = 0.01

try:
    user_input_iterations = int(input("Input Amount of Iterations: "))
    if user_input_iterations <= 0: 
        print("Cannot be 0 or negative. Default values are set")
        user_input_iterations = 100
except (TypeError, ValueError):
    print("Input was invalid. Default values are set")
    user_input_iterations = 100

randomize_check = input("Randomize dataset? [Y/N]: ")
if randomize_check in ['Y', 'y']:
    np.random.shuffle(dataset)

# Data splitting
# Split features and labels
featureset = dataset[:, 4:-1] # ? Skips first 4 columns as per MC01 Specs
labelset = dataset[:, -1] # ? Only takes last column aka labels
# Split the dataset for training and testing
X_train, X_test, y_train, y_test = Split_Data(featureset, labelset, test_size=user_input_test_size)

# Create a perceptron
perceptron = Perceptron(num_features=X_train.shape[1])
# * Train the perceptron
perceptron.train(X_train, y_train, learning_rate=user_learning_rate,iter=user_input_iterations)

Accuracy_Test(X_test)


if(input("\nCompare against KNN? [Y/N]:") in ['y', 'Y']):
    from KNN import KNN
    knn = KNN()
    knn_pred = knn.predict(X_train, y_train, X_test, k = int(input("Please enter k:")))
    knn.accuracy(y_true = y_test, y_pred = knn_pred)
else: 
    pass