from gpt4all import GPT4All




class LLM():
    def __init__(self, model_name = None, model_path = None) -> None:
        self.gpt = GPT4All(model_name = model_name, model_path = model_path)

        self.user = "### Instruction" #"USER"
        self.assistant = "### Response" #"ASSISTANT"
        self.input = "### Input"
        self.streaming = False

        # self.context = lambda question: f""" # OLD FORMAT
        # By considering above input memories from me, answer the question if its provided in memory, else just answer without memory: {question}
        # """ # additional linking context

        self.context = lambda question: f"""
        By considering below input memories from me, answer the question if its provided in memory, else just answer without memory: {question}
        """ # additional linking context

    def response(self, request):
        # return self.gpt.generate(prompt = f"{self.user}: {request}\n{self.assistant}: ", streaming = self.streaming)
        return self.gpt.generate(prompt = f"{self.user}:\n{request}\n{self.assistant}:\n", streaming = self.streaming)

    # def memory_response(self, request, memory_queries): # OLD FORMAT
    #     queries = f"{self.user}:\n"

    #     for i, query in enumerate(memory_queries):
    #         # queries += f"{self.user}: {request}\n{self.assistant}: {query}"
    #         queries += f"MEMORY CHUNK {i}: {query}\n"

    #     # queries += f"{self.user}: {self.context(request)}\n{self.assistant}: "
    #     # queries += f"{self.context(request)}\n{self.assistant}: "
    #     queries += f"{self.context(request)}\n{self.assistant}:\n"

    #     # print(queries)

    #     return self.gpt.generate(prompt = queries, streaming = self.streaming)

    def memory_response(self, request, memory_queries):
        queries = f"{self.user}:\n{self.context(request)}\n{self.input}:\n"

        for i, query in enumerate(memory_queries):
            # queries += f"{self.user}: {request}\n{self.assistant}: {query}"
            queries += f"MEMORY CHUNK {i}: {query}\n"

        queries += f"{self.assistant}:\n"

        return self.gpt.generate(prompt = queries, streaming = self.streaming)


