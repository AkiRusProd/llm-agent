from llm_agent import LLMAgent
from llm import LlamaCPPLLM, GPT4AllLLM
from embedder import Embedder
from search_engine import SearchEngine
from summarizer import Summarizer
from query_db import CollectionOperator
from dotenv import dotenv_values

env = dotenv_values(".env")

def chat_gpt4all():
    llm_agent.llm.streaming = True
    system_template = 'A chat between a curious user and an artificial intelligence assistant.'

    with llm_agent.llm.gpt.chat_session(system_template):
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
    system_template = '<<SYS>>A chat between a curious user and an artificial intelligence assistant.<</SYS>>'

    llm_agent.llm.gpt.eval(llm_agent.llm.gpt.tokenize(system_template.encode("utf-8")))

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

    """Keep in mind that user and helper tokens may vary between LLMs."""
    llm.user = "### Instruction" #"USER"
    llm.assistant = "### Response" #"ASSISTANT"
    
    embedder = Embedder()
    search_engine = SearchEngine()
    summarizer = Summarizer()

    total_memory_co = CollectionOperator("total-memory", embedder = embedder)

    llm_agent = LLMAgent(llm, total_memory_co, summarizer, search_engine, use_summarizer = False)

    chat()