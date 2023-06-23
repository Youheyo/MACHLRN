import numpy as np
import pandas as pd

data = pd.read_csv("dataset/kNN Internet Survey Sheet.csv")

data = data.iloc[1:, 4:] #Skip First Row and First 4 columns as per specs

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

