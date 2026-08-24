import time

import numpy as np

from vector_db.ivf import IVFIndex



NUM_VECTORS = 10_000
DIMENSIONS = 32
NUM_CLUSTERS = 100



vectors_array = np.random.rand(
    NUM_VECTORS,
    DIMENSIONS
).astype(np.float32)


# Convert to dictionary.
vectors = {
    str(i): vectors_array[i]
    for i in range(NUM_VECTORS)
}


print("Building IVF index...")

index = IVFIndex(
    n_clusters=NUM_CLUSTERS
)

start = time.perf_counter()

index.build(vectors)

build_time = (
    time.perf_counter() - start
)

print(
    f"Index build time: "
    f"{build_time:.4f}s"
)


query = np.random.rand(
    DIMENSIONS
).astype(np.float32)


for nprobe in [1, 5, 10, 25, 50, 100]:

    start = time.perf_counter()

    results = index.search(
        query=query,
        vectors=vectors,
        k=10,
        nprobe=nprobe
    )

    elapsed = (
        time.perf_counter() - start
    )

    print(
        f"nprobe={nprobe:3d} "
        f"| time={elapsed:.6f}s"
    )