from llm import LLM
from query_db import CollectionOperator
from summarizer import Summarizer


llm = LLM("D:\\Code\\GPTS\\nous-hermes-13b.ggmlv3.q4_0.bin")
collection_operator = CollectionOperator("queries")
summarizer = Summarizer()



class LTMAgent():
    def __init__(self, llm: LLM = None, qdb: CollectionOperator = None, add_memory = True) -> None:
        self.llm = llm
        self.qdb = qdb
        self.memory_access_threshold = 1.5
        self.top_k = 3
        self.add_memory = add_memory

        self.summarizer = summarizer

    def response(self, request):
        memory_queries_data = self.qdb.query(request, n_results = self.top_k, return_text = False)
        memory_queries = memory_queries_data['documents'][0]
        memory_queries_distances = memory_queries_data['distances'][0]

        acceptable_memory_queries = []

        for query, distance in list(zip(memory_queries, memory_queries_distances)):
            if distance < self.memory_access_threshold:
                acceptable_memory_queries.append(query)

        if len(acceptable_memory_queries) > 0:
            llm_response = self.llm.memory_response(request, acceptable_memory_queries)
        else:
            llm_response =  list(self.llm.response(request))

        if self.add_memory and len(acceptable_memory_queries) == 0:     
            summary = self.summarizer(f"{self.llm.user}:\n{request}\n{self.llm.assistant}:\n{''.join(llm_response)}")

            self.qdb.add(summary)
            
        return llm_response

    

ltm_agent = LTMAgent(llm, collection_operator, add_memory = True)

# print(ltmgpt.response("What is the student name?"))
# print(ltmgpt.response("What is the student`s interests?"))
# print(ltmgpt.response("How old is this student?"))