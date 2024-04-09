import re
from search_engine import SearchEngine
from summarizer import Summarizer
from db import DBInstance
from guidance.models import LlamaCpp
from guidance import gen, select
from utils import logging


enable_logging = True





class LLMAgent:
    def __init__(
        self,
        model_name: str = None,
        db_instance: DBInstance = None,
        summarizer: Summarizer = None,
        search_engine: SearchEngine = None,
        use_summarizer=True,
    ) -> None:

        self.llm = LlamaCpp(model=model_name, n_ctx=8192, verbose=True)
        self.db_instance = db_instance
        self.memory_access_threshold = 1.5
        # self.similarity_threshold = 0.5 # [0; 1]
        self.db_n_results = 3
        self.se_n_results = 1
        self.use_summarizer = use_summarizer

        self.summarizer = summarizer
        self.search_engine = search_engine
        self.chat_prompt_template = "[INST] {prompt} [/INST]"

    @logging(enable_logging, message="[Adding to memory]")
    def add(self, request):
        self.db_instance.add(request) if request != "" else None

    @logging(enable_logging, message="[Querying memory]")
    def memory_response(self, request):
        memory_queries_data = self.db_instance.query(
            request, n_results=self.db_n_results, return_text=False
        )
        memory_queries = memory_queries_data["documents"][0]
        memory_queries_distances = memory_queries_data["distances"][0]

        acceptable_memory_queries = []

        for query, distance in list(zip(memory_queries, memory_queries_distances)):
            if distance < self.memory_access_threshold:
                # if (1 - distance) >= self.similarity_threshold:
                acceptable_memory_queries.append(query)

        if len(acceptable_memory_queries) == 0:
            # return self.llm.response(request)
            return None

        prompt_template = """\
        By considering below input memories from me, answer the question if its provided in memory, else just answer without memory:
        QUESTION:
        `{text}`
        MEMORY CHUNKS:{context}
        """

        context = ""
        for i, query in enumerate(memory_queries):
            context += f"MEMORY CHUNK {i}: {query}\n"

        queries = prompt_template.format(text=request, context=context)

        out = (
            self.llm
            + self.chat_prompt_template.format(prompt=queries)
            + " "
            + gen(name="response", temperature=1)
        )
        return out["response"]

    @logging(enable_logging, message="[Searching]")
    def search(self, request):
        search_response = self.search_engine.search(
            request, n_results=self.se_n_results
        )

        for response in search_response:
            response["content"] = self._summarize(response["content"])

        prompt_template = """\
        You have been given access to the Internet.
        By considering below search results, summarize the information if its provided in search result, else just answer without search results:
        QUESTION:
        `{text}`
        SEARCH RESULTS:
        {context}
        """

        context = ""
        for i, query in enumerate(search_response):
            context += f"SEARCH TITLE: {query['title']}\nSEARCH LINK: {query['link']}\nSEARCH CONTENT: {query['content']}\n"

        queries = prompt_template.format(text=request, context=context)
        out = (
            self.llm
            + self.chat_prompt_template.format(prompt=queries)
            + " "
            + gen(name="response", temperature=1)
        )
        return out["response"]

    @logging(enable_logging, message="[Summarizing]", color="green")
    def _summarize(self, text, min_length=30, max_length=100):
        return self.summarizer(text, min_length, max_length)

    @logging(enable_logging, message="[Extracting question]", color="green")
    def _extract_query(self, request):
        prompt_template = """\
        Extract the question from the following text:
        "{request}"
        """

        prompt = prompt_template.format(request=request)

        out = (
            self.llm
            + self.chat_prompt_template.format(prompt=prompt)
            + " "
            + f"""Extracted question: "{gen(name='question', temperature=1, stop='"')}"""
        )

        return out["question"]

    @logging(enable_logging, message="[Response]")
    def response(self, request):
        out = (
            self.llm
            + self.chat_prompt_template.format(prompt=request)
            + " "
            + gen(name="response", temperature=1)
        )
        return out["response"]

    def generate(self, request: str):
        choises = ["ANSWER", "WEB_SEARCH", "DB_SEARCH", "ADD_MEMORY"]

        prompt_template = """\
        {request}
        Please choose an option from below:
        ANSWER - Answer the question
        WEB_SEARCH - Search on the web
        DB_SEARCH - Find information in the database
        ADD_MEMORY - Add information to the database
        Default option: ANSWER
        """

        prompt = prompt_template.format(request=request)

        out = (
            self.llm
            + self.chat_prompt_template.format(prompt=prompt)
            + " "
            + f"Choice: {select(choises, name='choice')}"
        )

        if out["choice"] == "ANSWER":
            return self.response(request)

        elif out["choice"] == "WEB_SEARCH":
            return self.search(self._extract_query(request))

        elif out["choice"] == "DB_SEARCH":
            return self.memory_response(self._extract_query(request))

        elif out["choice"] == "ADD_MEMORY":
            matches = re.findall(r'"(.*?)"', request)

            for item in matches:
                self.add(item)

            return "Done!"
