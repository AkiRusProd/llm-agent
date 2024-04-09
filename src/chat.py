from llm_agent import LLMAgent
from embedder import HFEmbedder
from search_engine import SearchEngine
from summarizer import Summarizer
from db import DBInstance
from dotenv import dotenv_values

env = dotenv_values(".env")


def chat():
    while True:
        user_text_request = input("You > ")

        bot_text_response = llm_agent.generate(user_text_request)
        print(f"Bot < {bot_text_response}")


if __name__ == "__main__":
    embedder = HFEmbedder()
    search_engine = SearchEngine()
    summarizer = Summarizer()

    db_instance = DBInstance("long-term-memory", embedder=embedder)

    llm_agent = LLMAgent(
        env["LLM_PATH"], db_instance, summarizer, search_engine, use_summarizer=False
    )

    chat()
