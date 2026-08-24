import numpy as np

from vector_db.kmeans import kmeans


vectors = np.array([
    [1, 1],
    [2, 1],

    [8, 8],
    [9, 8],

    [5, 5]
])


centroids, labels = kmeans(
    vectors,
    k=2,
    iterations=10
)


print("Centroids:")
print(centroids)

print("\nLabels:")
print(labels)