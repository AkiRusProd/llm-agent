import chromadb
import uuid
from embedder import Embedder



class CollectionOperator():
    def __init__(self, collection_name, db_path = "src/db/"):
        self.embedder = Embedder()
        self.client = chromadb.PersistentClient(path = db_path)
        self.collection = self.client.get_or_create_collection(name = collection_name, embedding_function = self.embedder.get_embeddings)

    def add(self, text, metadata = {}):
        self.collection.add(
            documents = [text],
            # metadatas = [metadata],
            ids = [str(uuid.uuid4())]
        )

    def query(self, query, n_results, return_text = True):
        query = self.collection.query(
            query_texts = query,
            n_results = n_results,
        )

        if return_text:
            return query['documents'][0]
        else:
            return query


collection_operator = CollectionOperator("queries")

# examples of memory queries:
# collection_operator.add("Rustam Akimov, computer science student who enjoys programming in his free time. He`s age is 20 years old.")
# collection_operator.add("Technologies used by Rustam Akimov: numpy, pandas, pytorch, docker, git, sql, linux etc.")
# collection_operator.add("Rustam`s interests: playing guitar, watching movies, listening to music, reading books, etc.")

# print(collection_operator.query("What is the student name?", 2))
