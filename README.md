# long-term-memory-llm
LLM using long-term memory through vector database

## Description
This repository enables the language model to use long-term memory through a vector database. The application is built using [gpt4all nous-hermes-13b llm](https://gpt4all.io/index.html) and [chromadb](https://github.com/chroma-core/chroma).


### Examples:
Information stored in vector database:
```
[
  "Rustam Akimov, a 20 years old computer science student who enjoys programming in his free time.,"    
  "Technologies used by Rustam Akimov: numpy, pandas, pytorch, docker, git, sql, linux etc.,"    
  "Rustam`s interests: playing guitar, watching movies, listening to music, reading books, etc."
]
```

User questions:
```
[
  "What is the student name?"    
  "What is the student`s interests?"    
  "How old is this student?"    
]
```

Model answers:
```
[
    "The student's name is Rustam Akimov."    
    "Based on the information provided, it appears that Rustam Akimov's interests include playing guitar, watching movies, listening to music, reading books, and programming in his free time."    
    "Based on the information provided in the memory statements, we can determine that Rustam Akimov's age is 20 years old."
]
```

TODO:   
Add forgetting mechanism
