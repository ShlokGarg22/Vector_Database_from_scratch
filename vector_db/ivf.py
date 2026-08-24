import numpy as np

from .kmeans import kmeans
from .similarity import cosine_similarity


class IVFIndex:

    def __init__(self, n_clusters=10):
        """
        Create an empty IVF index.

        n_clusters:
            Number of clusters we want to create.

            Example:
                n_clusters = 10

            means K-Means will divide our vectors
            into approximately 10 groups.
        """

        self.n_clusters = n_clusters

        self.centroids = None

        # Maps:
        #
        # cluster_id -> vector IDs
        #
        # Example:
        #
        # {
        #     0: ["doc1", "doc4"],
        #     1: ["doc2", "doc3"]
        # }
        #
        # This is the "inverted file" part.
        self.clusters = {}

    def build(self, vectors):
        """
        Build the IVF index.

        Parameters
        ----------
        vectors:
            Dictionary:

                {
                    "doc1": [0.1, 0.2],
                    "doc2": [0.8, 0.9]
                }

        What happens:

            1. Run K-Means.
            2. Get cluster assignments.
            3. Create a mapping from cluster -> vector IDs.
        """


        vector_ids = list(vectors.keys())


        vector_values = np.array(
            list(vectors.values())
        )

        # -----------------------------------------
        # STEP 3: Run K-Means
        # -----------------------------------------
        #
        # K-Means returns:
        #
        # centroids
        #     -> center of every cluster
        #
        # labels
        #     -> cluster assigned to every vector
        #
        # Example:
        #
        # vector IDs:
        # [doc1, doc2, doc3, doc4]
        #
        # labels:
        # [0,    0,    1,    1]
        #
        # meaning:
        #
        # doc1 -> cluster 0
        # doc2 -> cluster 0
        # doc3 -> cluster 1
        # doc4 -> cluster 1

        self.centroids, labels = kmeans(
            vector_values,
            k=self.n_clusters,
            iterations=20
        )


        self.clusters = {
            cluster_id: []
            for cluster_id in range(self.n_clusters)
        }


        for vector_id, cluster_id in zip(
            vector_ids,
            labels
        ):

            self.clusters[
                int(cluster_id)
            ].append(vector_id)

    def search(
        self,
        query,
        vectors,
        k=5,
        nprobe=1
    ):
        """
        Search the IVF index.

        nprobe:
            Number of closest clusters to search.

            nprobe=1:
                Search only the closest cluster.

            nprobe=5:
                Search the 5 closest clusters.

            Higher nprobe generally means:
                better recall
                slower search
        """

        # Convert query into NumPy array.
        query = np.array(query)


        distances = np.linalg.norm(
            self.centroids - query,
            axis=1
        )


        nearest_clusters = np.argsort(
            distances
        )[:nprobe]


        candidate_ids = []

        for cluster_id in nearest_clusters:

            candidate_ids.extend(
                self.clusters[int(cluster_id)]
            )


        results = []

        for vector_id in candidate_ids:

            vector = vectors[vector_id]

            score = cosine_similarity(
                query,
                vector
            )

            results.append(
                (vector_id, score)
            )


        results.sort(
            key=lambda x: x[1],
            reverse=True
        )
        
        return results[:k]