from llm import LLM
from query_db import CollectionOperator


llm = LLM("D:\\Code\\GPTS\\nous-hermes-13b.ggmlv3.q4_0.bin")
collection_operator = CollectionOperator("queries")



class LTMGPT():
    def __init__(self, llm: LLM = None, qdb: CollectionOperator = None) -> None:
        self.llm = llm
        self.qdb = qdb
        self.memory_access_threshold = 1.5

    def response(self, request):
        memory_queries_data = self.qdb.query(request, n_results = 5, return_text = False)
        memory_queries = memory_queries_data['documents'][0]
        memory_queries_distances = memory_queries_data['distances'][0]

        acceptable_memory_queries = []

        for query, distance in list(zip(memory_queries, memory_queries_distances)):
            if distance < 1.5:
                acceptable_memory_queries.append(query)

        if len(acceptable_memory_queries) > 0:
            return self.llm.memory_response(request, acceptable_memory_queries)
        else:
            return self.llm.response(request)

    

ltmgpt = LTMGPT(llm, collection_operator)

# print(ltmgpt.response("What is the student name?"))
# print(ltmgpt.response("What is the student`s interests?"))
# print(ltmgpt.response("How old is this student?"))