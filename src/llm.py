from gpt4all import GPT4All




class LLM():
    def __init__(self, model_name = None, model_path = None) -> None:
        self.gpt = GPT4All(model_name = model_name, model_path = model_path)

        self.user = "USER"
        self.assistant = "ASSISTANT"
        self.streaming = False

        self.context = lambda question: f"""
        By considering above input memories from me, answer the question if its provided in memory, else just anser without memory: {question}
        """ # additional linking context

    def response(self, request):
        return self.gpt.generate(prompt = f"{self.user}: {request}\n{self.assistant}: ", streaming = self.streaming)

    def memory_response(self, request, memory_queries):
        queries = f"{self.user}:\n"

        for i, query in enumerate(memory_queries):
            # queries += f"{self.user}: {request}\n{self.assistant}: {query}"
            queries += f"MEMORY CHUNK {i}: {query}\n"

        # queries += f"{self.user}: {self.context(request)}\n{self.assistant}: "
        queries += f"{self.context(request)}\n{self.assistant}: "

        # print(queries)

        return self.gpt.generate(prompt = queries, streaming = self.streaming)


