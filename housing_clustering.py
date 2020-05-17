import matplotlib.pyplot as plt
from housing_preprocessing import full_untransformed_data, corr_index_names, lasso_index_names, unscaled_y
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn import metrics

# Graph the untransformed features
untransformed_best_lasso=full_untransformed_data[lasso_index_names]
untransformed_best_corr=full_untransformed_data[corr_index_names]

# Common features between the measures
plt.figure(figsize=(15,10))
plt.scatter(untransformed_best_lasso.loc[:, "TotalInsideArea"], untransformed_best_lasso.loc[:, "GrLivArea"])
plt.title('Group Living Area vs Total Inside Areas')
plt.show()

# Lasso exclusive features
plt.scatter(untransformed_best_lasso.loc[:, "TotalInsideArea"], untransformed_best_lasso.loc[:, "BsmtFinSF1"])
plt.scatter(untransformed_best_lasso.loc[:, "GrLivArea"], untransformed_best_lasso.loc[:, "BsmtFinSF1"])

# Correlation exclusive features
plt.scatter(untransformed_best_corr.loc[:, "TotalInsideArea"], untransformed_best_corr.loc[:, "GarageArea"])
plt.scatter(untransformed_best_corr.loc[:, "GrLivArea"], untransformed_best_corr.loc[:, "GarageArea"])

# Combine the best features from Lasso and Correlation methods of feature selection
best_of_both=pd.concat([untransformed_best_lasso, untransformed_best_corr], axis=1)
best_features=best_of_both.loc[:,~best_of_both.columns.duplicated()]

# Plot any other common numerical features
plt.scatter(best_of_both.loc[:, "GarageArea"], best_of_both.loc[:, "BsmtFinSF1"])

# Scale the highest features according to Correlation
standardized_corr=pd.DataFrame(StandardScaler().fit_transform(untransformed_best_corr), columns=untransformed_best_corr.columns)

# Fit the PCA models
pca_best=PCA(n_components=standardized_corr.shape[1])
components_pca=pca_best.fit_transform(standardized_corr)
feature_count=range(1,pca_best.n_components_+1)
pca_var=pca_best.explained_variance_ratio_.tolist()

# PCA Best Components
print(pca_best.components_)

# Check the variance ratio for each principal component
plt.figure(figsize=(15,10))
plt.bar(feature_count, pca_var, color="blue")
plt.title("PCA Explained Variance by component for Feature Selection")
plt.ylabel("Variance Explained")
plt.xlabel("Principal Component Number")
plt.show()

# Choose the first 4 components
pca_clusters_corr=pd.DataFrame(components_pca[:, :4])

# Visualize the components to see if there are any clusters
fig, axes=plt.subplots(3, 2, figsize=(15,15))
axes[0, 0].scatter(pca_clusters_corr.iloc[:,0], pca_clusters_corr.iloc[:, 1])
axes[0, 0].set_title('PC 2 vs PC 1')
axes[0, 1].scatter(pca_clusters_corr.iloc[:,0], pca_clusters_corr.iloc[:, 2])
axes[0, 1].set_title('PC 3 vs PC 1')
axes[1, 0].scatter(pca_clusters_corr.iloc[:,0], pca_clusters_corr.iloc[:, 3])
axes[1, 0].set_title('PC 4 vs PC 1')
axes[1, 1].scatter(pca_clusters_corr.iloc[:,1], pca_clusters_corr.iloc[:, 2])
axes[1, 1].set_title('PC 3 vs PC 2')
axes[2, 0].scatter(pca_clusters_corr.iloc[:,1], pca_clusters_corr.iloc[:, 3])
axes[2, 0].set_title('PC 4 vs PC 2')
axes[2, 1].scatter(pca_clusters_corr.iloc[:,2], pca_clusters_corr.iloc[:, 3])
axes[2, 1].set_title('PC 4 vs PC 3')

# Plot the original y values
val=0.
plt.plot(unscaled_y, np.zeros_like(unscaled_y)+ val, 'x')
plt.xlab("Housing Prices")
plt.show()

# Grouping bins of Sales Prices
print(unscaled_y.describe())
values_db=[unscaled_y.describe()[1]- unscaled_y.describe()[2], unscaled_y.describe()[1], unscaled_y.describe()[1] + unscaled_y.describe()[2],unscaled_y.describe()[1] + 2*unscaled_y.describe()[2]]
values_hier=[unscaled_y.describe()[4], unscaled_y.describe()[5], unscaled_y.describe()[6]]

y_db=unscaled_y.copy().to_frame()
y_hier=unscaled_y.copy().to_frame()
y_db["Label"]=0
y_hier["Label"]=0

# Group for DBScan results
obs_count=y_db.shape[0]
for i in range(obs_count):
    val=y_db.iloc[i, 0]
    if val <= values_db[0]:
        pass
    elif val <= values_db[1]:
        y_db.iloc[i,1]=1
    elif val <= values_db[2]:
        y_db.iloc[i,1]=2
    else:
        y_db.iloc[i,1]=3
        
# Group for Hierarchical clustering
for i in range(obs_count):
    val=y_hier.iloc[i, 0]
    if val <= values_hier[0]:
        pass
    elif val <= values_hier[1]:
        y_hier.iloc[i,1]=1
    else:
        y_hier.iloc[i,1]=2
        
# DBScan original y-values
fig, ax = plt.subplots()
db_scatter=ax.scatter(pca_clusters_corr.iloc[:,0], pca_clusters_corr.iloc[:, 1], c=y_db["Label"])
legend_1 = ax.legend(*db_scatter.legend_elements(), loc="lower right", title="Values")
plt.show()

# Density based clustering for PC 1 and PC 2
dbscan=DBSCAN(eps = 0.375, min_samples=15)
dbscan_clustering=dbscan.fit(pca_clusters_corr.iloc[:,:2])
db_labels=dbscan_clustering.labels_
core_points=np.zeros_like(db_labels, dtype= bool)
core_points[dbscan.core_sample_indices_] = True
n_clusters=len(set(db_labels))-(1 if -1 in db_labels else 0)

print('Estimated number of clusters: %d' % n_clusters)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(pca_clusters_corr.iloc[:,:2], db_labels))

# Use black for noise points
unique_labels = set(db_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (db_labels == k)

    xy = pca_clusters_corr[class_member_mask & core_points]
    plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = pca_clusters_corr[class_member_mask & ~core_points]
    plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters)
plt.show()

# Choose correlation based features for Hierarchical Clustering
plt.figure(figsize=(15,10))
dendrogram=sch.dendrogram(sch.linkage(standardized_corr, method="ward"))
plt.title('Dendrogram of Housing Price Correlation features')
plt.ylabel('Euclidean distances')
plt.show()

# We will use Agglomerative Clustering to recursively merge individual points into our desired cluster
AggCluster = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage ='ward')
Agg_predict=AggCluster.fit_predict(standardized_corr)

hierarchical_best_features=standardized_corr.copy()
hierarchical_best_features['Label']=Agg_predict

# Scatterplot of labels according to hierarchical clustering
hierarch_features=[]
no_of_clusters=3
for i in range(no_of_clusters):
    hierarch_features.append(hierarchical_best_features[hierarchical_best_features['Label']==i])

colors=["red", "blue", "orange", "purple", "yellow"]
plt.figure(figsize=(15,10))
for i in range(no_of_clusters):
    plt.scatter(hierarch_features[i].loc[:, 'GrLivArea'], hierarch_features[i].loc[:, 'TotalInsideArea'], c=colors[i])

plt.xlabel("Group Living Area")
plt.ylabel("Total Inside Area")
plt.title("Inside Area vs Group Living Area")
plt.show()

# Overall Quality vs Group Living Area
plt.figure(figsize=(15,10))
for i in range(no_of_clusters):
    plt.scatter(hierarch_features[i].loc[:, 'TotalInsideArea'], hierarch_features[i].loc[:, 'OverallQual'], c=colors[i])

plt.xlabel("Group Living Area")
plt.ylabel("Overall Quality")
plt.title("Inside Area vs Quality")
plt.show()

plt.scatter(standardized_corr.loc[:, 'TotalInsideArea'], standardized_corr.loc[:, 'OverallQual'], c=y_hier["Label"])

# Overall Quality vs Garage Area
plt.figure(figsize=(15,10))
for i in range(no_of_clusters):
    plt.scatter(hierarch_features[i].loc[:, 'GarageArea'], hierarch_features[i].loc[:, 'OverallQual'], c=colors[i])

plt.xlabel("Garage Area")
plt.ylabel("Overall Quality")
plt.title("Quality vs Garage Area")
plt.show()

plt.scatter(standardized_corr.loc[:, 'GarageArea'], standardized_corr.loc[:, 'OverallQual'], c=y_hier["Label"])

# Bathrooms vs Group Living Area
plt.figure(figsize=(15,10))
for i in range(no_of_clusters):
    plt.scatter(hierarch_features[i].loc[:, 'TotalInsideArea'], hierarch_features[i].loc[:, 'Bathrooms'], c=colors[i])

plt.xlabel("Group Living Area")
plt.ylabel("Bathrooms")
plt.title("Bathrooms vs Inside Area")
plt.show()

plt.scatter(standardized_corr.loc[:, 'TotalInsideArea'], standardized_corr.loc[:, 'Bathrooms'], c=y_hier["Label"])
