import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
%matplotlib inline

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


data = pd.read_csv('/content/drive/My Drive/datasets/AppleWatch.csv')
print(data.shape)
print(data.head())

data['Calories'] = np.where(data['Calories'].between(0.7,1), 1, data['Calories'])
data['Calories'].astype(int)

data["Steps"] = data["Steps"].div(data["Distance"])

data["Steps"].mul(100)

data['Steps'].replace(to_replace ="0",  value ="1", inplace = True)

data["Steps"].astype(int)
data['Distance'] = np.where(data['Distance'].between(0,1000), 1, data['Distance'])
print(data['Distance'].head())
data['Distance'].astype(int)

data['Age'] = np.where(data['Age'].between(21,25), 1, data['Age'])
data['Age'] = np.where(data['Age'].between(26,30), 2, data['Age'])

data['Age'] = np.where(data['Age'].between(31,35), 3, data['Age'])

data['Age'] = np.where(data['Age'].between(36,40), 4, data['Age'])

data['Age'] = np.where(data['Age'].between(41,44), 5, data['Age'])
print(data['Age'].head())
data['Age'].astype(int)

data['Gender'].replace(to_replace ="M",  value ="2", inplace = True)
data['Gender'].replace(to_replace ="F",  value ="1", inplace = True)

print(data['Gender'].head())
data['Gender'].astype(int)

data['Weight'] = np.where(data['Weight'].between(110,150), 1, data['Weight'])
data['Weight'] = np.where(data['Weight'].between(151,200), 2, data['Weight'])
data['Weight'] = np.where(data['Weight'].between(201,250), 3, data['Weight'])

data['Weight'].astype(int)

data['Height'] = np.where(data['Height'].between(5.1,5.5), 1, data['Height'])
data['Height'] = np.where(data['Height'].between(5.6,6), 2, data['Height'])
data['Height'] = np.where(data['Height'].between(6.1,6.7), 3, data['Height'])


data['Height'].astype(int)

data['Activity'].replace(to_replace ="0.Sleep",  value ="0", inplace = True)
data['Activity'].replace(to_replace ="1.Sedentary",  value ="1", inplace = True)
data['Activity'].replace(to_replace ="2.Light",  value ="2", inplace = True)
data['Activity'].replace(to_replace ="3.Moderate",  value ="3", inplace = True)
data['Activity'].replace(to_replace ="4.Vigorous",  value ="4", inplace = True)

data['Activity'].astype(int)

data['Heart'] = np.where(data['Heart'].between(0,119), 1, data['Heart'])
data['Heart'] = np.where(data['Heart'].between(120,200), 2, data['Heart'])

print(data['Heart'].head())

data['Heart'].astype(int)
data.isnull()

print(data.shape)
print(data.head())

from sklearn.cluster import KMeans

# Number of clusters
kmeans = KMeans(n_clusters=2)
# Fitting the input data
kmeans = kmeans.fit(data)
# Getting the cluster labels
labels = kmeans.predict(data)
# Centroid values
centroids = kmeans.cluster_centers_

# Comparing with scikit-learn centroids
print(labels) # From Scratch
print(centroids) # From sci-kit learn
kmeans.labels_
sample_test=np.array([-3.0,-3.0,0.1,0.2,-.3,.3,.2,.1,.2])
second_test=sample_test.reshape(1, -1)
kmeans.predict(second_test)
