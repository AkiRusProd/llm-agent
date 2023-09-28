from llm import LLM
from query_db import CollectionOperator
from search_engine import SearchEngine
from summarizer import Summarizer
from embedder import Embedder
from utils import logging
from dotenv import dotenv_values
env = dotenv_values(".env")

search_engine = SearchEngine()
summarizer = Summarizer()
embedder = Embedder()



llm = LLM(env['GPT4ALL_LLM'])
total_memory_co = CollectionOperator("total-memory", embedder = embedder)





enable_logging = True




class LLMAgent():
    def __init__(
        self, 
        llm: LLM = None, 
        tm_qdb: CollectionOperator = None, 
        summarizer: Summarizer = None, 
        search_engine: SearchEngine = None,
        use_summarizer = True,
       
    ) -> None:

        self.llm = llm
        self.tm_qdb = tm_qdb
        self.memory_access_threshold = 1.5
        # self.similarity_threshold = 0.5 # [0; 1]
        self.db_n_results = 3
        self.se_n_results = 3
        self.use_summarizer = use_summarizer
       
        self.summarizer = summarizer
        self.search_engine = search_engine
       

    @logging(enable_logging, message = "[Adding to memory]")
    def add(self, request):
        # summary = self.summarizer(f"{self.llm.user}:\n{request}\n{self.llm.assistant}:\n{''.join(response)}")
        
        summary = self.summarize(request) if self.use_summarizer else request

        self.tm_qdb.add(summary) if summary != "" else None

        response = self.llm.response(request)

        return response

    @logging(enable_logging, message = "[Querying memory]")
    def memory_response(self, request):
        memory_queries_data = self.tm_qdb.query(request, n_results = self.db_n_results, return_text = False)
        memory_queries = memory_queries_data['documents'][0]
        memory_queries_distances = memory_queries_data['distances'][0]

        acceptable_memory_queries = []

        for query, distance in list(zip(memory_queries, memory_queries_distances)):
            if distance < self.memory_access_threshold:
            # if (1 - distance) >= self.similarity_threshold:
                acceptable_memory_queries.append(query)

        if len(acceptable_memory_queries) > 0:
            response = self.llm.memory_response(request, acceptable_memory_queries)
        else:
            response = self.llm.response(request)

        return response

    @logging(enable_logging, message = "[Searching]")
    def search(self, request):
        search_response = self.search_engine.search(request, n_results = self.se_n_results)

        for response in search_response:
            response['content'] = self.summarize(response['content'])

        return self.llm.search_response(request, search_response)

    @logging(enable_logging, message = "[Summarizing]", color = "green")
    def summarize(self, text, min_length = 30, max_length = 100):
        return self.summarizer(text, min_length, max_length)


    @logging(enable_logging, message = "[Response]")
    def response(self, request):
        return self.llm.response(request)

    
    def generate(self, request: str):
        if request.upper().startswith("MEM"):
            response = self.memory_response(request[len("MEM"):])
        elif request.upper().startswith("REMEM"): #and len(acceptable_memory_queries) == 0
            response = self.add(request[len("REMEM"):])
        elif request.upper().startswith("WEB"):
            response = self.search(request[len("WEB"):])
        else:
            response = self.response(request)
            
        return response

    

llm_agent = LLMAgent(llm, total_memory_co, summarizer, search_engine, use_summarizer = False)
