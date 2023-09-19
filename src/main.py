from llm import LLM
from query_db import CollectionOperator


llm = LLM("D:\\Code\\GPTS\\nous-hermes-13b.ggmlv3.q4_0.bin")
collection_operator = CollectionOperator("queries")



class LTMGPT():
    def __init__(self, llm: LLM = None, qdb: CollectionOperator = None) -> None:
        self.llm = llm
        self.qdb = qdb

    def response(self, request):
        memory_queries = self.qdb.query(request, n_results = 5)
        
        return self.llm.response(request, memory_queries)

    

ltmgpt = LTMGPT(llm, collection_operator)

print(ltmgpt.response("What is the student name?"))
print(ltmgpt.response("What is the student`s interests?"))
print(ltmgpt.response("How old is this student?"))