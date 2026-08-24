import time

import numpy as np

from vector_db import VectorDB


def benchmark(num_vectors, dimensions=128):

    print(
        f"\nBenchmarking {num_vectors:,} vectors "
        f"with {dimensions} dimensions..."
    )

    db = VectorDB(
        storage_path=f"data/benchmark_{num_vectors}.json"
    )

    vectors = np.random.rand(
        num_vectors,
        dimensions
    ).tolist()

    for i, vector in enumerate(vectors):

        db.vectors[str(i)] = {
            "vector": vector,
            "metadata": {}
        }

    query = np.random.rand(dimensions).tolist()

    # Start timer.
    start = time.perf_counter()

    # Search.
    results = db.search(
        query,
        k=10
    )

    # Stop timer.
    end = time.perf_counter()

    elapsed = end - start

    print(f"Search time: {elapsed:.4f} seconds")

    return elapsed


if __name__ == "__main__":

    benchmark(1_000)

    benchmark(10_000)

    benchmark(100_000)