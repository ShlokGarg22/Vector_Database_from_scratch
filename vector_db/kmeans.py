import numpy as np

def kmeans(vectors,k,iterations = 10):
    """
    grp vectors into k clusters
    """
    vectors = np.array(vectors)

    random_indices = np.random.choice(
        len(vectors),
        size = k,
        replace=False
    )
    centroids = vectors[random_indices].copy()

    print("initial centroids")


    for _ in range(iterations):

        distances = np.linalg.norm(
            vectors[:,None,:] - centroids[None,:,:],
            axis=2
        )

        labels = np.argmin(
            distances,axis=1
        )
        for cluster_id in range(k):

            cluster_vectors = vectors[
                labels == cluster_id
            ]

            if len(cluster_vectors) > 0:

                centroids[cluster_id] = np.mean(
                    cluster_vectors,
                    axis=0
                )

    return centroids, labels       