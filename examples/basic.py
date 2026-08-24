# from vector_db import VectorDB
# from vector_db.similarity import cosine_similarity

# db = VectorDB()

# db.insert("doc1",[0.1,0.2,0.3])
# db.insert("doc2",[0.8,0.2,0.5])
# db.insert("doc3",[0.3,0.4,0.7])
# db.insert("doc4",[-1,-2,-3])


# query = [1,2,3]

# print("doc1",cosine_similarity(query,db.vectors["doc1"]))
# print("doc2",cosine_similarity(query,db.vectors["doc2"]))
# print("doc3",cosine_similarity(query,db.vectors["doc3"]))
# print("doc4",cosine_similarity(query,db.vectors["doc4"]))

# results = db.search(query,k=3)

# print(db.vectors)
# print(results)
from vector_db import VectorDB
from vector_db.ivf import IVFIndex

db = VectorDB(
    storage_path="data/ivf_demo.json"
)


db.insert(
    "doc1",
    [1, 1],
    {
        "category": "AI"
    }
)

db.insert(
    "doc2",
    [2, 1],
    {
        "category": "AI"
    }
)

db.insert(
    "doc3",
    [8, 8],
    {
        "category": "ML"
    }
)

db.insert(
    "doc4",
    [9, 8],
    {
        "category": "ML"
    }
)

db.insert(
    "doc5",
    [5, 5],
    {
        "category": "ML"
    }
)


vectors = {
    vector_id: data["vector"]
    for vector_id, data in db.vectors.items()
}

index = IVFIndex(
    n_clusters=2
)


index.build(vectors)


print("\n========== CENTROIDS ==========")

print(index.centroids)



print("\n========== CLUSTERS ==========")

print(index.clusters)


query = [8, 8]


results = index.search(
    query=query,
    vectors=vectors,
    k=2
)


print("\n========== SEARCH RESULTS ==========")

for vector_id, score in results:

    print(
        f"ID: {vector_id} | "
        f"Similarity: {score:.4f}"
    )