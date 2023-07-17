import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_BMU(SOM, x):
    distSq = (np.square(SOM - x)).sum(axis = 0)
    return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

def update_weights(SOM, X_Train, learning_rate, radius, BMU_coord, step=3):
    g, h = BMU_coord

    if radius < 1e-3:
        SOM[g,h,:] += learning_rate * (X_Train - SOM[g,h,:])
        return SOM
    
    for i in range(max(0, h-step) , min(SOM.shape[0], g+step)):
        for j in range(max(0, h-step) , min(SOM.shape[1], g+step)):
            dist_sq = np.square(i - g) + np.square(j - h)
            dist_func = np.exp(-dist_sq / 2 / radius)
            SOM[i, j, :] += learning_rate * dist_func * (X_Train - SOM[i, j, :])
    return SOM

""" 
TODO Cycle Points are at 50,000 and 75,000. Total cycles is at 100,000
?       Starts at 0 Cycles
TODO Learning rate starts at .5, .25, .1
TODO Radius starts at 3, 2, 1
"""
def train_SOM(SOM, X_train, learning_rate = .5, radius = 3, lr_decay = .1, radius_decay = .1, iter = 10):
    for iter in range(iter):
        #rand.shuffle(X_train)
        print("Iteration:", iter)
        for x in X_train:
            g, h, = find_BMU(SOM, x)
            SOM = update_weights(SOM, x, learning_rate, radius, (g,h))

        learning_rate = learning_rate * np.exp(-iter * lr_decay)
        radius = radius * np.exp(-iter * radius_decay)
    return SOM

dataset = pd.read_csv('dataset/MCO1 New InternetSurveyDataset.csv')
dataset = dataset.iloc[:, :-1].values.astype(float) # ? Everything but risk taker tag

#Grid Dimension
grid = 16 # 16x16 grid

weights = np.random.randint(0,1, dataset.shape[1])

SOM = train_SOM(dataset, weights, iter=5)
print(SOM)