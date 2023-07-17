import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ! ChatGPT generated

def initialize_weights(input_dim, output_dim):
    return np.random.rand(input_dim, output_dim)


def find_best_matching_unit(input_vector, weights):
    input_tile = np.tile(input_vector, (weights.shape[0], 1))
    distances = np.sum((input_vector - weights) ** 2, axis=1)
    return np.argmin(distances)


def update_weights(input_vector, weights, bmu, learning_rate, neighborhood_radius):
    distance_squares = np.sum((np.indices(weights.shape) - bmu.reshape(-1, 1, 1)) ** 2, axis=0)
    influence = np.exp(-(distance_squares) / (2 * neighborhood_radius ** 2))
    weights += learning_rate * influence.reshape(-1, 1) * (input_vector - weights)


def train_som(data, output_dim, num_epochs, learning_rate, initial_neighborhood_radius):
    input_dim = data.shape[1]
    weights = initialize_weights(input_dim, output_dim)

    for epoch in range(num_epochs):
        for input_vector in data:
            bmu = find_best_matching_unit(input_vector, weights)
            update_weights(input_vector, weights, bmu, learning_rate, initial_neighborhood_radius)

    return weights


# Load the dataset using pandas
dataset = pd.read_csv('dataset/MCO1 New InternetSurveyDataset.csv')
dataset = dataset.iloc[:, :-1].values.astype(float) # Convert DataFrame to numpy array

# Normalize the data (optional but often helpful for SOMs)
data = (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)

# Set the parameters for the SOM
output_dim = 16  # Output grid dimensions (e.g., 10x10)
num_epochs = 100  # Number of training epochs
learning_rate = 0.1  # Learning rate
initial_neighborhood_radius = max(output_dim // 2, 1)  # Initial neighborhood radius

# Train the SOM
weights = train_som(data, output_dim, num_epochs, learning_rate, initial_neighborhood_radius)

# Visualize the SOM grid
grid_shape = (output_dim, output_dim)
grid_positions = np.indices(grid_shape).reshape(2, -1).T
grid_weights = weights.reshape(-1, data.shape[1])
plt.scatter(grid_positions[:, 0], grid_positions[:, 1])
plt.scatter(grid_positions[:, 0], grid_positions[:, 1], c=grid_weights)
plt.colorbar()
plt.show()