import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

dataset = pd.read_csv('dataset/MCO1 New InternetSurveyDataset.csv')
dataset = dataset.values.astype(float)

def adjust_time(time):
	if time > 86400:
		time /= 86400
		time_unit = "days"
	elif time > 3600:
		time /= 3600
		time_unit = "hours"
	elif time > 60:
		time /= 60
		time_unit = "minutes"
	else:
		time_unit = "seconds"
	return time, time_unit


class SOM:
	def __init__(self, size, feature_dim, iter, radius = 3, learning_rate = .5):
		self.size = size
		self.feature_dim = feature_dim
		self.iterations = iter
		self.weights = np.random.rand(size, size, feature_dim)
		self.radius = radius
		self.learning_rate = learning_rate

	def find_bmu(self, data):
		distance = np.linalg.norm(self.weights - data, axis = 2)
		return np.unravel_index(np.argmin(distance, axis=None), distance.shape)

	def update_weights(self, data, bmu):
		for x in range(self.size):
			for y in range(self.size):
				dist_to_bmu = np.linalg.norm(np.subtract(bmu, (x, y)))

				influence = np.exp(-(dist_to_bmu**2) / (2 * (self.radius**2))) * self.learning_rate

				self.weights[x, y] += influence * (data - self.weights[x, y])

	def train(self, data, debug = False):
		#region Debug Time Start
		if debug is True:
			st = time.time()
			total_predicted_time = 0
			#ETA = []
			print("Som Training has started")
		#endregion

		#region Iteration Algorithm
		for iter in range(self.iterations):
			
			#region Start of Iteraton Time tracking
			if debug is True:
				print("Current Iteration:", iter, "/", self.iterations)
				iterStartTime = time.time()
			#endregion
			#region Parameter adjustment on cycle
			if iter > 75000:
				self.learning_rate = 0.1
				self.radius = 1
			elif iter > 50000:
				self.learning_rate = 0.25
				self.radius = 2
			#endregion
			#region Algorithm Proper
			for row in data:
				bmu = self.find_bmu(row)

				self.update_weights(row, bmu)
			#endregion
			#region Debug Time Increments
			if debug is True:
				iterEndTime = time.time()
				# ? ETA of a single Runtime
				#ETA.append(iterEndTime - iterStartTime)
				total_predicted_time += (iterEndTime - iterStartTime)
				# ? Average Runtime
				#avg_eta = sum(ETA) / len(ETA)
				avg_eta = total_predicted_time / (iter+1)
				# ? Predicted ETA from Average runtime
				predicted_eta = (self.iterations - iter) * avg_eta

				predicted_eta, time_unit = adjust_time(predicted_eta)
				print("Estimated Finish time:", round(predicted_eta, 2), time_unit)
			#endregion
		#endregion

		#region Debug End Statistics
		if debug is True:
			et = time.time()
			total_time, time_unit= adjust_time(et - st)
			print("Code ran for", round(total_time,2), time_unit)
			print("Avg Runtime per iteration:", round(avg_eta , 2))
		#endregion


def KMeans(data, k = 5):
	kmeans_data = som.weights.reshape(-1, data.shape[1])
	centroids = kmeans_data[np.random.choice(kmeans_data.shape[0], k, replace = False)]

	# TODO Get the Data (Gender, Age, Income, Rural/Urban, Risktaker Tag) from clusters
	# TODO Display percentages of each cluster
	# TODO Display Global Percentage afterwards

	for _ in range(100):
		distances = np.linalg.norm(kmeans_data[:, np.newaxis] - centroids, axis = 2)
		labels = np.argmin(distances, axis = 1)

		for i in range(k):
			centroids[i] = np.mean(kmeans_data[labels == i], axis = 0)

	plt.figure(figsize=(8, 8))
	plt.scatter(kmeans_data[:, 0], kmeans_data[:, 1], c=labels, cmap='viridis')
	plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
	plt.title("K-means Clustering")
	plt.xlabel("Feature 1")
	plt.ylabel("Feature 2")
	plt.show()

	#plt.show()

som = SOM(16, dataset.shape[1], iter = 5)
som.train(dataset, debug = True)

KMeans(dataset, k = 5)

plt.show()

