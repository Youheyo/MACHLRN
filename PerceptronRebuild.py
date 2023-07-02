import numpy as np
import pandas as pd

# Load the dataset
dataset = pd.read_csv('dataset/kNN Internet Survey Sheet.csv')

# Remove the column labels from the first row
dataset = dataset.iloc[1:]

# Split the dataset into training and testing sets
training_data = dataset[103:].values.astype(float)
testing_data = dataset[:100].values.astype(float)

# Extract the features and labels from the training data
X_train = training_data[:, :-3]
y_train = training_data[:, -3]

# Extract the features and labels from the testing data
X_test = testing_data[:, :-3]
y_test = testing_data[:, -4]

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
        for _ in range(num_epochs):
            for i, _ in enumerate(X):
                x = X[i]
                y_true = y[i]
                y_pred = self.predict(x)
                # Update weights and bias based on prediction error
                self.weights += learning_rate * (y_true - y_pred) * x
                self.bias += learning_rate * (y_true - y_pred)

# Create a perceptron model
perceptron = Perceptron(num_features=X_train.shape[1])

# Train the model
perceptron.train(X_train, y_train)

# Test the model

def accuracy(X_test):
    accuracy = 0
    for i, test_data in enumerate(X_test):
        x = X_test[i]
        y_true = y_test[i]
        y_pred = perceptron.predict(x)

        if y_pred == y_true:
            accuracy += 1

    accuracy /= len(X_test)
    return accuracy

print("Accuracy:", accuracy(X_test))
