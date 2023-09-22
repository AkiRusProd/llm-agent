from llm import LLM
from query_db import CollectionOperator
from summarizer import Summarizer
from embedder import Embedder
from utils import logging


summarizer = Summarizer()
embedder = Embedder()


llm = LLM("D:\\Code\\GPTS\\nous-hermes-13b.ggmlv3.q4_0.bin")
tm_collection_operator = CollectionOperator("total-memory", embedder = embedder)
rm_collection_operator = CollectionOperator("recent-memory", embedder = embedder) # TODO: add recent memory




enable_logging = True




class LLMAgent():
    def __init__(
        self, 
        llm: LLM = None, 
        tm_qdb: CollectionOperator = None, 
        rm_qdb: CollectionOperator = None, 
        summarizer: Summarizer = None, 
        use_summarizer = True,
    ) -> None:

        self.llm = llm
        self.tm_qdb = tm_qdb
        self.rm_qdb = rm_qdb
        self.memory_access_threshold = 1.5
        self.top_k = 3
        self.use_summarizer = use_summarizer

        self.summarizer = summarizer
       

    @logging(enable_logging, message = "[Adding to memory]")
    def add(self, request):
        # summary = self.summarizer(f"{self.llm.user}:\n{request}\n{self.llm.assistant}:\n{''.join(llm_response)}")
        
        summary = self.summarizer(request) if self.use_summarizer else request

        self.tm_qdb.add(summary) if summary != "" else None

        llm_response = self.llm.response(request)

        return llm_response

    @logging(enable_logging, message = "[Querying memory]")
    def memory_response(self, request):
        memory_queries_data = self.tm_qdb.query(request, n_results = self.top_k, return_text = False)
        memory_queries = memory_queries_data['documents'][0]
        memory_queries_distances = memory_queries_data['distances'][0]

        acceptable_memory_queries = []

        for query, distance in list(zip(memory_queries, memory_queries_distances)):
            if distance < self.memory_access_threshold:
                acceptable_memory_queries.append(query)

        if len(acceptable_memory_queries) > 0:
            llm_response = self.llm.memory_response(request, acceptable_memory_queries)
        else:
            llm_response = self.llm.response(request)

        return llm_response

    @logging(enable_logging, message = "[Searching]")
    def search(self, request):
        pass
    
    @logging(enable_logging, message = "[Response]")
    def response(self, request):
        return self.llm.response(request)

    
    def generate(self, request: str):
        if request.upper().startswith("MEM"):
            llm_response = self.memory_response(request[len("MEM"):])
        elif request.upper().startswith("REMEM"): #and len(acceptable_memory_queries) == 0
            llm_response = self.add(request[len("REMEM"):])
        elif request.upper().startswith("WEB"):
            llm_response = self.search(request[len("WEB"):])
        else:
            llm_response = self.response(request)
            
        return llm_response

    

llm_agent = LLMAgent(llm, tm_collection_operator, rm_collection_operator, summarizer, use_summarizer = False)
