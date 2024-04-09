# long-term-memory-llm
RAG-based LLM using long-term memory through vector database        

## Description
This repository enables the large language model to use long-term memory through a vector database (This method is called RAG (Retrieval Augmented Generation) — this is a technique that allows LLM to retrieve facts from an external database). The application is built with [mistral-7b-instruct-v0.2.Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) (using [LLAMA_cpp_python](https://github.com/abetlen/llama-cpp-python) binding) and [chromadb](https://github.com/chroma-core/chroma). User can ask in natural language to add information to db, find information from db or the Internet using [guidance](https://github.com/guidance-ai/guidance).


### Current features:
- add new memory: add information (in quotes) in natural language to the database
- query memory: request information from a database in natural language
- web search (experimental): find information from the Internet in natural language

### Diagram:
![Diagram](images/llm-agent.png)

### Example:
```
You > Hi
LOG: [Response]
Bot < Hello! How can I assist you today?
You > Please add information to db "The user name is Rustam Akimov"
LOG: [Adding to memory]
Bot < Done!
You > Can you find on the Internet who is Pavel Durov
LOG: [Extracting question]
LOG: [Searching]
LOG: [Summarizing]
Bot < According to the search results provided, Pavel Durov is a Russian entrepreneur who co-founded Telegram Messenger Inc.
You > Please find information in db who is Rustam Akimov
LOG: [Extracting question]
LOG: [Querying memory]
Bot < According to the input memories, your name is Rustam Akimov.
```

### Usage:
- Install requirements.txt
- Download [mistral-7b-instruct-v0.2.Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) (Note: you can use other models)
- Get [Google API key](https://developers.google.com/webmaster-tools/search-console-api/v1/configure) and [Search Engine ID](https://programmablesearchengine.google.com/controlpanel/create)
- Specify variables in .env
- Run [chat.py](src/chat.py)
