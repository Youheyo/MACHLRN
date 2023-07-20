#.venv/Scripts/activate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans

dataset = pd.read_csv('dataset/MCO1 New InternetSurveyDataset.csv')

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

	def train(self, data):

		data = data.values
		cap1 = cap2 = False
		print("SOM training has started")
		print(f"SOM specs: {self.size}x{self.size} grid, {self.iterations} iterations, Starts at {self.radius} radius and {self.learning_rate * 100}% learning rate")

		#region Iteration Algorithm
		for iter in tqdm(range(self.iterations), desc="Training SOM"):
			np.random.shuffle(data)
			#region Parameter adjustment on cycle
			if iter > 75000 and cap2 is False:
				self.learning_rate = 0.1
				self.radius = 1
				cap2 = True
				tqdm.write(f"75,000 Iterations reached. Adjusting learning rate to {self.learning_rate * 100}, Radius to {self.radius}")
			elif iter > 50000 and cap1 is False:
				self.learning_rate = 0.25
				self.radius = 2
				cap1 = True
				tqdm.write(f"50,000 Iterations reached. Adjusting learning rate to {self.learning_rate * 100}, Radius to {self.radius}")
			#endregion

			#region Algorithm Proper
			# ? Loops through the whole dataset causing it to be longer
			# for row in data:
			# 	bmu = self.find_bmu(row)

			# 	self.update_weights(row, bmu)

			# ? Randomizes and loops on a single row of data
			row = data[np.random.randint(0, data.shape[0])]

			bmu = self.find_bmu(row)

			self.update_weights(row,bmu)

			#endregion

		#endregion

		print("SOM training Complete!")


def KMeans_Cluster(data, k = 5, iter = 100):
	kmeans_data = som.weights.reshape(-1, data.shape[1])
	centroids = kmeans_data[np.random.choice(kmeans_data.shape[0], k, replace = False)]

	for _ in range(iter):
		distances = np.linalg.norm(kmeans_data[:, np.newaxis] - centroids, axis = 2)
		labels = np.argmin(distances, axis = 1)

		for i in range(k):
			centroids[i] = np.mean(kmeans_data[labels == i], axis = 0)
	
	return labels, centroids

def plot(labels, centroids, data):
	data = som.weights.reshape(-1, data.shape[1])
	# Create a DataFrame to store the cluster labels and feature values
	cluster_df = pd.DataFrame({'Cluster': labels})
	cluster_df['Gender'] = dataset['a1']
	cluster_df['Age Range'] = dataset['a2']
	cluster_df['Income'] = dataset['a3']
	cluster_df['Rural/Urban'] = dataset['a4']
	cluster_df['Risk Taker'] = dataset['RiskTaker']

	total_data = np.zeros(5)

	for i in range(5):
	# Group the DataFrame by cluster and calculate the percentage of features 1 and 2
		cluster_data = cluster_df.loc[cluster_df['Cluster'] == i]
		cluster_size = len(cluster_data)
		gender = round(cluster_data['Gender'].value_counts(sort = False) / cluster_size * 100, 2) 
		age = round(cluster_data['Age Range'].value_counts(sort = False) / cluster_size * 100, 2)
		income = round(cluster_data['Income'].value_counts(sort = False)  / cluster_size * 100, 2)
		rural = round(cluster_data['Rural/Urban'].value_counts(sort = False) / cluster_size * 100, 2)
		rt = round(cluster_data['Risk Taker'].value_counts(sort = False) / cluster_size * 100, 2)

		print(f"Cluster {i+1}")
		print(f"{gender[0]}% Male \t\t {gender[1]} Female ")
		print(f"{age[0]}% ages 9 - 11 \t {age[1]} ages 12 - 17 ")
		print(f"{income[0]}% Low Income \t {income[1]} Mid/High Income ")
		print(f"{rural[0]}% Rural \t\t {rural[1]} Urban ")
		print(f"{rt[0]}% non-Risk-taker \t {rt[1]} Risk-taker\n")

		figure, ax = plt.subplots(2, 3)
		plt.suptitle(f"Cluster {i+1}")
		ax[0, 0].pie([gender[0], gender[1]], labels=["Male", "Female"])
		ax[0, 0].set_title("Gender")

		ax[1, 0].pie([age[0], age[1]], labels=["9 - 11", "12 - 17"])
		ax[1, 0].set_title("Age Range")

		ax[0, 1].pie([income[0], income[1]], labels=["Low", "Mid/High"])
		ax[0, 1].set_title("Income")

		ax[1, 1].pie([rural[0], rural[1]], labels=["Rural", "Urban"])
		ax[1, 1].set_title("Rural/Urban")

		ax[0, 2].pie([rt[0], rt[1]], labels=["Non", "Risktaker"])
		ax[0, 2].set_title("Risk Taker")

		figure.delaxes(ax[1,2])

		total_data[0] += gender[0]
		total_data[1] += age[0]
		total_data[2] += income[0]
		total_data[3] += rural[0]
		total_data[4] += rt[0]

	for i in range(total_data.size):
		total_data[i] /= 500
		total_data[i] = round(total_data[i]* 100, 2)

	print(f"Global Data")
	print(f"{total_data[0]}% Male \t\t {round(100 - total_data[0],2)} Female ")
	print(f"{total_data[1]}% ages 9 - 11 \t {round(100 - total_data[1],2)} ages 12 - 17 ")
	print(f"{total_data[2]}% Low Income \t {round(100 - total_data[2],2)} Mid/High Income ")
	print(f"{total_data[3]}% Rural \t\t {round(100 - total_data[3],2)} Urban ")
	print(f"{total_data[4]}% non-Risk-taker \t {round(100 - total_data[4],2)} Risk-taker\n")


	figure, ax = plt.subplots(2, 3)

	figure.suptitle("Global Stats")
	ax[0, 0].pie([total_data[0], 100 - total_data[0]], labels=["Male", "Female"])
	ax[0, 0].set_title("Gender")

	ax[1, 0].pie([total_data[1], 100 - total_data[1]], labels=["9 - 11", "12 - 17"])
	ax[1, 0].set_title("Age Range")

	ax[0, 1].pie([total_data[2], 100 - total_data[2]], labels=["Low", "Mid/High"])
	ax[0, 1].set_title("Income")

	ax[1, 1].pie([total_data[3], 100 - total_data[3]], labels=["Rural", "Urban"])
	ax[1, 1].set_title("Rural/Urban")

	ax[0, 2].pie([total_data[4], 100 - total_data[4]], labels=["Non", "Risktaker"])
	ax[0, 2].set_title("Risk Taker")

	figure.delaxes(ax[1,2])

	# plt.figure(figsize=(8, 8))
	# plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
	# plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
	# plt.title("K-means Clustering")
	plt.show()

som = SOM(16, dataset.shape[1], iter = 100)
som.train(dataset)

labels, centroids = KMeans_Cluster(dataset, k = 5, iter = 100)
plot(labels, centroids, dataset)

# ! Comparing with sklearn knn remove befoer submission along side import (line 7)
# * It looks similar btw
kmeans = KMeans(n_clusters = 5, max_iter=100)
labels = kmeans.fit_predict(som.weights.reshape(-1, som.feature_dim))

plot(labels, kmeans.cluster_centers_, dataset)