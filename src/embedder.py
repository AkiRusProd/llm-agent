from gpt4all import Embed4All


class Embedder():
    def __init__(self):
        self.embedder = Embed4All() # default: all-MiniLM-L6-v2

    def get_embeddings(self, texts):
        if type(texts) == str:
            texts = [texts]
        
        embedddings = []
        for text in texts:
            embedddings.append(self.embedder.embed(text))

        return embedddings

    def __call__(self, text):
        return self.get_embeddings(text)

