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


db = VectorDB()

db.insert(
    "doc1",
    [1, 2, 3],
    {
        "category": "AI",
        "source": "ai_notes.pdf"
    }
)

db.insert(
    "doc2",
    [2, 4, 6],
    {
        "category": "programming",
        "source": "python.pdf"
    }
)

db.insert(
    "doc3",
    [1, 2, 2],
    {
        "category": "AI",
        "source": "cnn.pdf"
    }
)


query = [1, 2, 3]


results = db.search(
    query,
    k=5,
    filters={"does_not_exist": "whatever"}
)


for result in results:
    print(result)