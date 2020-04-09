from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from housingprediction import standardize_full_numerical, y
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns

# Fit the PCA model
pca=PCA(n_components=20)
components=pca.fit_transform(standardize_full_numerical)
feature_count=range(pca.n_components_)
pca_var=pca.explained_variance_ratio_.tolist()

# Check the variance ratio for each principle component
plt.bar(feature_count, pca_var, color="blue")

# Visualize the components to see if there are any clusters
pca_clusters=pd.DataFrame(components)

plt.figure(figsize=(20,10))
plots = []
count=0
for i in range(3):
    for j in range(4):
        if j>i:
            count+=1
            if count <=3:
                ax = plt.subplot2grid((5,4), (1,count))
                ax.scatter(pca_clusters[i], pca_clusters[j])
            else:
                ax = plt.subplot2grid((5,4), (2,count-3))
                ax.scatter(pca_clusters[i], pca_clusters[j])
plt.show()

# Minimize inertia by evaluating KMeans at various k values
cluster_inertia=[]
for i in range(1,11):
    k_clusters=KMeans(n_clusters=i, init='k-means++', max_iter=1000, n_init=20, random_state=42)
    k_clusters.fit(pca_clusters.iloc[:, :4])
    cluster_inertia.append(k_clusters.inertia_)

# Visualize the inertia function over clusters    
plt.plot(range(1,11), cluster_inertia)
plt.title("Inertia per cluster")
plt.xlabel("Cluster")
plt.ylabel("Inertia")
plt.show()

# Delta between increasing the cluster size by 1
delta_inertia=[(cluster_inertia[i-1]-cluster_inertia[i]) for i in range(1,len(cluster_inertia))]
plt.plot(range(1,len(delta_inertia)+1), delta_inertia)
plt.title("Change in Inertia per cluster added")
plt.xlabel("Cluster k+1")
plt.ylabel("Decline by adding cluster")
plt.show()

# Choose the value where the decrease in inertia declines
k_means=KMeans(n_clusters=4, init='k-means++', max_iter=1000, n_init=20, random_state=42)
k_means.fit(pca_clusters.iloc[:, :4])
groupings=k_means.predict(pca_clusters.iloc[:, :4])
centers=k_means.cluster_centers_

# Predicted clusters
pca_clusters["Cluster"]=k_means.labels_
cluster_centers=pd.DataFrame(centers, columns=["C1", "C2", "C3", "C4"])

# Choose k clusters
clusters=pca_clusters.iloc[:, :4]
clusters["Cluster"]=k_means.labels_

# Graph the clusters
colors=["red", "blue", "green"]
sns.pairplot(clusters, hue="Cluster", palette="Dark2", diag_kind="kde" ,size=1.85)
plt.show()



