# long-term-memory-llm
LLM using long-term memory through vector database

## Description
This repository enables the language model to use long-term memory through a vector database. The application is built using [gpt4all nous-hermes-13b llm](https://gpt4all.io/index.html) and [chromadb](https://github.com/chroma-core/chroma).


### Example:
```
You > Hi
LOG: [Response]
Bot < Hello! How can I assist you today?
You > web who is Pavel Durov
LOG: [Searching]
Bot < According to the search results provided, Pavel Durov is a Russian entrepreneur who co-founded Telegram Messenger Inc. He was also involved in developing The Open Network (TON), but later withdrew from the project due to litigation with the US Securities and Exchange Commission (SEC).
You > mem who is Rustam Akimov
LOG: [Querying memory]
Bot < According to the input memories, your name is Rustam Akimov.
```

TODO:   
Add forgetting mechanism
