from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from housingprediction import standardize_full_numerical, standardize_best_features,y
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np

# Fit the PCA model
pca=PCA(n_components=9)
components=pca.fit_transform(standardize_best_features)
feature_count=range(pca.n_components_)
pca_var=pca.explained_variance_ratio_.tolist()

# Check the variance ratio for each principal component
plt.figure(figsize=(15,10))
plt.bar(feature_count, pca_var, color="blue")
plt.title("PCA Explained Variance by component")
plt.ylabel("Variance Explained")
plt.xlabel("Principal Component Number")
plt.show()

# Choose the first 4 components
pca_clusters=pd.DataFrame(components[:, :4])

# Visualize the components to see if there are any clusters
fig, axes=plt.subplots(3, 2, figsize=(15,15))
axes[0, 0].scatter(pca_clusters.iloc[:,0], pca_clusters.iloc[:, 1])
axes[0, 0].set_title('PC 2 vs PC 1')
axes[0, 1].scatter(pca_clusters.iloc[:,0], pca_clusters.iloc[:, 2])
axes[0, 1].set_title('PC 3 vs PC 1')
axes[1, 0].scatter(pca_clusters.iloc[:,0], pca_clusters.iloc[:, 3])
axes[1, 0].set_title('PC 4 vs PC 1')
axes[1, 1].scatter(pca_clusters.iloc[:,1], pca_clusters.iloc[:, 2])
axes[1, 1].set_title('PC 3 vs PC 2')
axes[2, 0].scatter(pca_clusters.iloc[:,1], pca_clusters.iloc[:, 3])
axes[2, 0].set_title('PC 4 vs PC 2')
axes[2, 1].scatter(pca_clusters.iloc[:,2], pca_clusters.iloc[:, 3])
axes[2, 1].set_title('PC 4 vs PC 3')

# Minimize SSE by evaluating KMeans at various k values
cluster_inertia=[]
for i in range(1,21):
    k_clusters=KMeans(n_clusters=i, init='k-means++', max_iter=1000, n_init=20, random_state=42)
    k_clusters.fit(pca_clusters)
    cluster_inertia.append(k_clusters.inertia_)

# Visualize the inertia function over clusters  
plt.figure(figsize=(15,10))
plt.plot(range(1,21), cluster_inertia)
plt.title("Inertia per cluster")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Cluster")
plt.ylabel("Inertia")
plt.show()

# Difference between increasing the cluster size by 1
delta_inertia=[(cluster_inertia[i-1]-cluster_inertia[i]) for i in range(1,len(cluster_inertia))]
plt.figure(figsize=(15,10))
plt.plot(range(1,len(delta_inertia)+1), delta_inertia)
plt.title("Change in Inertia per cluster added")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Previous Cluster k")
plt.ylabel("Decline by adding cluster")
plt.show()

# Choose the value where the decrease in inertia declines
k_means=KMeans(n_clusters=6, init='k-means++', max_iter=1000, n_init=20, random_state=42)
k_means.fit(pca_clusters)
groupings=k_means.predict(pca_clusters)
centers=k_means.cluster_centers_

# Choose k clusters
clusters=pca_clusters
clusters["Cluster"]=k_means.labels_

# Graph the clusters
colors=["red", "blue", "green", "orange", "violet", "black", "yellow"]
sns.pairplot(clusters, hue="Cluster", palette="Dark2", diag_kind="kde" , height=1.85)
plt.show()



