import chromadb
import uuid
import datetime
from embedder import BaseEmbedder, HFEmbedder
from dotenv import dotenv_values

env = dotenv_values(".env")
DB_PATH = env["DB_PATH"]


class DBInstance():
    def __init__(self, collection_name, db_path = DB_PATH, embedder: BaseEmbedder = None):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path = db_path)
        self.collection = self.client.get_or_create_collection(name = collection_name, embedding_function = self.embedder)

    def add(self, text, metadata = {}):
        metadata['timestamp'] = str(datetime.datetime.now())

        self.collection.add(
            documents = [text],
            metadatas = [metadata],
            ids = [str(uuid.uuid4())]
        )

    def delete(self, id):
        self.collection.delete(id)

    def query(self, query, n_results, return_text = True):
        query = self.collection.query(
            query_texts = query,
            n_results = n_results,
        )

        if return_text:
            return query['documents'][0]
        else:
            return query
