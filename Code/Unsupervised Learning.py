from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas  as pd
import numpy as np

# Load the dataset
wholesale_data = pd.read_csv("C:\\Users\\17789\\LHL\\ml-project-unsupervised-learning\\Wholesale_Data.csv", index_col=0)

# Handling outliers by capping at the 95th percentile
for column in wholesale_data.columns:
    threshold = np.percentile(wholesale_data[column], 95)
    wholesale_data[column] = np.where(wholesale_data[column] > threshold, threshold, wholesale_data[column])

# Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(wholesale_data)

# K-means Clustering Preparation
inertia = []
range_values = range(1, 10)
for i in range_values:
    kmeans = KMeans(n_clusters=i, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Dimensionality Reduction with PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Preparing data for visualization
pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])

# Results
outlier_handled_data = wholesale_data.copy()
print(inertia)
print(pca_df.head())
