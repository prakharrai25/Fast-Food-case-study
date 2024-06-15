import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data

mcdonalds = pd.read_csv('/content/mcdonalds.csv')

# Preprocess data
MD_x = mcdonalds.iloc[:, 0:11]
MD_x = (MD_x == "Yes").astype(int)

# Compute column means
col_means = np.round(MD_x.mean(), 2)
print(col_means)

# PCA
MD_pca = PCA()
MD_pca.fit(MD_x)
MD_pca_summary = pd.DataFrame({
    'Standard deviation': np.round(MD_pca.explained_variance_, 4),
    'Proportion of Variance': np.round(MD_pca.explained_variance_ratio_, 4),
    'Cumulative Proportion': np.round(np.cumsum(MD_pca.explained_variance_ratio_), 4)
})
print(MD_pca_summary)

# Plot PCA
plt.plot(MD_pca_summary.index + 1, MD_pca_summary['Proportion of Variance'], 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance')
plt.title('Importance of components')
plt.show()

# K-means clustering
kmeans = KMeans(n_clusters=4, random_state=1234)
kmeans.fit(MD_x)
MD_k4 = kmeans.labels_

# Hierarchical clustering and plot
MD_vclust = PCA().fit_transform(MD_x.T)
MD_vclust_order = np.argsort(MD_vclust[:, 0])
MD_barchart_order = np.flip(MD_vclust_order)
plt.bar(np.arange(1, 5), MD_k4[MD_barchart_order], color='grey')
plt.xlabel('Number of segments')
plt.ylabel('Cluster Label')
plt.show()

# PCA and clustering plot
MD_pca_transformed = MD_pca.transform(MD_x)
MD_pca_df = pd.DataFrame(MD_pca_transformed, columns=['PC1', 'PC2'])
MD_pca_df['Cluster'] = MD_k4

plt.scatter(MD_pca_df['PC1'], MD_pca_df['PC2'], c=MD_pca_df['Cluster'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Mosaic plot
pd.crosstab(MD_k4, mcdonalds['Like']).plot(kind='bar', stacked=True, colormap='viridis')
plt.xlabel('Segment number')
plt.show()

pd.crosstab(MD_k4, mcdonalds['Gender']).plot(kind='bar', stacked=True, colormap='viridis')
plt.xlabel('Segment number')
plt.show()
