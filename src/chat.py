from llm_agent import LLMAgent
from llm import LlamaCPPLLM, GPT4AllLLM
from embedder import HFEmbedder, GPT4AllEmbedder
from search_engine import SearchEngine
from summarizer import Summarizer
from db import DBInstance
from dotenv import dotenv_values

env = dotenv_values(".env")

def chat_gpt4all():
    llm_agent.llm.streaming = True

    while True:
        user_text_request = input("You > ")

        bot_text_response = llm_agent.generate(user_text_request)
        
        if llm_agent.llm.streaming:
            print(f"Bot <", end = ' ')
            for token in bot_text_response:
                print(token, end = '')
            print()
        else:
            print(f"Bot < {bot_text_response}")

def chat_llama_cpp():
    llm_agent.llm.streaming = True
   
    while True:
        user_text_request = input("You > ")

        bot_text_response = llm_agent.generate(user_text_request)
        
        if llm_agent.llm.streaming:
            print(f"Bot <", end = ' ')
            for token in bot_text_response:
                print(token['choices'][0]['text'], end = '')
            print()
        else:
            print(f"Bot < {bot_text_response['choices'][0]['text']}")


if __name__ == "__main__":
    port_lib_name = "LLAMA_CPP"

    if port_lib_name == "LLAMA_CPP":
        LLM = LlamaCPPLLM
        chat = chat_llama_cpp
    else:
        LLM = GPT4AllLLM
        chat = chat_gpt4all

    llm = LLM(env['LLM_PATH'])
    
    embedder = HFEmbedder()
    search_engine = SearchEngine()
    summarizer = Summarizer()

    db_instance = DBInstance("long-term-memory", embedder = embedder)

    llm_agent = LLMAgent(llm, db_instance, summarizer, search_engine, use_summarizer = False)

    chat()