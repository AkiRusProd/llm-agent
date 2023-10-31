from typing import List, Optional, Any
from gpt4all import GPT4All
from llama_cpp import Llama

class BaseLLM():
    def __init__(self, model_name: Optional[str] = None, model_path: Optional[str] = None) -> None:
        self.user = "### Instruction" #"USER"
        self.assistant = "### Response" #"ASSISTANT"
        self.input = "### Input"
        self.streaming = False

        # additional linking contexts
        self.memory_context = lambda question: f"""
        By considering below input memories from me, answer the question if its provided in memory, else just answer without memory: {question}
        """ 

        self.search_context = lambda question: f"""
        You have been given access to the Internet. By considering below search results, summarize the information if its provided in search result, else just answer without search results: {question}
        """

    def generate(self, request: str, streaming: bool) -> Any:
        raise NotImplementedError

    def response(self, request: str) -> Any:
        return self.generate(f"{self.user}:\n{request}\n{self.assistant}:\n", streaming = self.streaming)

    def memory_response(self, request: str, memory_queries: List[str]) -> Any:
        queries = f"{self.user}:\n{self.memory_context(request)}\n{self.input}:\n"

        for i, query in enumerate(memory_queries):
            queries += f"MEMORY CHUNK {i}: {query}\n"

        queries += f"{self.assistant}:\n"

        return self.generate(queries, streaming = self.streaming)

    def search_response(self, request: str, search_results: List[dict[str, str, str]]) -> Any:
        queries = f"{self.user}:\n{self.search_context(request)}\n{self.input}:\n"

        for i, query in enumerate(search_results):
            queries += f"SEARCH TITLE: {query['title']}\nSEARCH LINK: {query['link']}\nSEARCH CONTENT: {query['content']}\n"

        queries += f"{self.assistant}:\n"

        return self.generate(queries, streaming = self.streaming)

class GPT4AllLLM(BaseLLM):
    def __init__(self, model_name: Optional[str] = None, model_path: Optional[str] = None) -> None:
        super().__init__(model_name, model_path)
        
        self.gpt = GPT4All(model_name = model_name, model_path = model_path, verbose=False)

    def generate(self, request: str, streaming: bool) -> Any:
        return self.gpt.generate(prompt = request, streaming = streaming)

class LlamaCPPLLM(BaseLLM):
    def __init__(self, model_name: Optional[str] = None) -> None:
        super().__init__(model_name)
        
        self.gpt = Llama(model_path = model_name, n_ctx=2048, verbose=False)

    def generate(self, request: str, streaming: bool) -> Any:
        return self.gpt.create_completion(prompt = request, stream = streaming, stop=[f"{self.user}:"])