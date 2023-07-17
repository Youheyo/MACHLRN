import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# ! It works but it is based on the code from ChatGPT.
# ! Further studying is needed in order to understand it myself
# * Extra details on specs
# TODO Cycle Points are at 50,000 and 75,000. Total cycles is at 100,000
# ?       Starts at 0 Cycles
# TODO Learning rate starts at .5, .25, .1
# TODO Radius starts at 3, 2, 1

st = time.time()

# Load the dataset using pandas
dataset = pd.read_csv('dataset/MCO1 New InternetSurveyDataset.csv')
data = dataset.values

# Define the SOM class
class SOM:
    def __init__(self, width, height, input_dim, epochs, radius = 3, learning_rate = .5):
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.epochs = epochs
        self.weights = np.random.rand(width, height, input_dim)
        self.radius = radius
        self.learning_rate = learning_rate

    def train(self, data):
        for epoch in range(self.epochs):
            if epoch > 75000:
                self.learning_rate = 0.1
                self.radius = 1
            elif epoch > 50000:
                self.learning_rate = 0.25
                self.radius = 2
                
            for sample in data:
                # Find the best matching unit (BMU)
                bmu = self.find_bmu(sample)

                # Update the weights of the BMU and its neighbors
                self.update_weights(sample, bmu)

    def find_bmu(self, sample):
        # Calculate the Euclidean distance between the sample and all neurons
        distances = np.linalg.norm(self.weights - sample, axis=2)

        # Find the neuron with the minimum distance (Best Matching Unit)
        bmu_coords = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_coords

    def update_weights(self, sample, bmu):

        for x in range(self.width):
            for y in range(self.height):
                # Calculate the Euclidean distance between the neuron and the BMU
                dist_to_bmu = np.linalg.norm(np.subtract(bmu, (x, y)))

                # Calculate the influence on the neuron based on the distance to BMU
                influence = np.exp(-(dist_to_bmu**2) / (2 * (self.radius**2))) * self.learning_rate

                # Update the weights of the neuron
                self.weights[x, y] += influence * (sample - self.weights[x, y])

# Create and train the SOM
som = SOM(16, 16, data.shape[1], epochs=100000)
som.train(data)

# Reshape the weights to 2D for K-means clustering
kmeans_data = som.weights.reshape(-1, data.shape[1])

# K-means clustering
k = 5  # Number of clusters
centroids = kmeans_data[np.random.choice(kmeans_data.shape[0], k, replace=False)]

for _ in range(100):
    # Assign samples to the nearest centroid
    distances = np.linalg.norm(kmeans_data[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    # Update the centroids
    for i in range(k):
        centroids[i] = np.mean(kmeans_data[labels == i], axis=0)

et = time.time()
print("Program took ", et - st, "seconds")

# Scatter plot visualization
plt.figure(figsize=(8, 8))
plt.scatter(kmeans_data[:, 0], kmeans_data[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.title("K-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()