
import torch
import os
import numpy as np
from transformers import AutoModel, AutoTokenizer
from chromadb import EmbeddingFunction
from gpt4all import Embed4All
from dotenv import dotenv_values

env = dotenv_values(".env")
os.environ['HUGGINGFACE_HUB_CACHE'] = env['HUGGINGFACE_HUB_CACHE']



class BaseEmbedder(EmbeddingFunction):
    def __init__(self):
        pass

    def get_embeddings(self, texts):
        raise NotImplementedError("Subclasses should implement this!")

    def __call__(self, text):
        return self.get_embeddings(text)


class GPT4AllEmbedder(BaseEmbedder):
    def __init__(self):
        self.embedder = Embed4All() # default: all-MiniLM-L6-v2

    def get_embeddings(self, texts):
        if type(texts) == str:
            texts = [texts]
        
        embeddings = []
        for text in texts:
            embeddings.append(self.embedder.embed(text))

        return embeddings

    def __call__(self, text):
        return self.get_embeddings(text)


# https://huggingface.co/princeton-nlp/sup-simcse-roberta-large
class HFEmbedder(BaseEmbedder):
    def __init__(self, model = 'princeton-nlp/sup-simcse-roberta-large'): #sentence-transformers/all-MiniLM-L6-v2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModel.from_pretrained(model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        

    def get_embeddings(self, texts):
        if type(texts) == str:
            texts = [texts]

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.detach().cpu().numpy()

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        return normalized_embeddings.tolist()

    def __call__(self, text):
        return self.get_embeddings(text)

