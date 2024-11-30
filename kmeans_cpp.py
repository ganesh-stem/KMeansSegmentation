import numpy as np
import kmeans
from sklearn.datasets import make_blobs
import time
import matplotlib.pyplot as plt

# Generate some random data
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Record start time
start_time = time.time()

# Create an instance of the KMeans class
kmeans_model = kmeans.KMeans(4, 400, 1e4, "lloyd", True)  # Pass arguments positionally

# Fit the KMeans model to the data
kmeans_model.fit(X)

# Get the labels assigned to each data point
labels = kmeans_model.getLabels()

# Get the centroids of the clusters
centroids = kmeans_model.getCentroids()

# Record end time
end_time = time.time()

# Calculate and print execution time
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

# Plot the data points and centroids
plt.figure(figsize=(8, 6))

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5, label='Data Points')

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='red', s=200, label='Centroids')

plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()