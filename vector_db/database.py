from .similarity import cosine_similarity

import json
import os
import heapq

class VectorDB:


    def __init__(self , storage_path="data/vectors.json"):

        # we will use a dicitionary to store our vectors where the key is vector id and value is actual vector
        self.storage_path = storage_path

        os.makedirs(
            os.path.dirname(self.storage_path),
            exist_ok=True
        )
        self.vectors={}
        self._load()

    
    def insert(self,vector_id,vector,metadata = None):
        #inserting with optional metadata

        if metadata is None:
            metadata = {}

        
        #stores in ram
        self.vectors[vector_id] = {
            "vector":vector,
            "metadata":metadata
        }

        #save the updated databse to the disk
        self._save()

    def search(self,query_vector,k=5,filters=None):


        if filters is None:
            filters = {}

        heap = []
        for vector_id , data in self.vectors.items():

            metadata = data["metadata"]

            matches_filter = True
            for key , expected_value in filters.items():
                actual_value = metadata.get(key)

                if actual_value != expected_value:
                    matches_filter = False
                    break
            if not matches_filter:
                continue
                
                   
            vector = data["vector"]

            score = cosine_similarity(query_vector,vector)

            item = (
                score , vector_id,metadata
            )

            if len(heap) < k:
                heapq.heappush(heap,item)

            elif score > heap[0][0]:
                heapq.heapreplace(heap,item)
            
            results = []

            for score, vector_id, metadata in heap:

                results.append({
                    "id": vector_id,
                    "score": score,
                    "metadata": metadata
                   })
        results.sort(
            key=lambda x: x["score"],
            reverse=True

        )
        return results[:k]
        
    def _save(self):
        """
        save the in-memory vectors to disk

        """
        with open(self.storage_path,"w") as file:

            json.dump(
                self.vectors,
                file
            )
    def _load(self):
        #load data from the disk if the database already exists

        if not os.path.exists(self.storage_path):
            return
        
        with open(self.storage_path,"r") as file:
            self.vectors = json.load(file)

