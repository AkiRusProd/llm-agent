from ltmgpt import ltmgpt



def chat():
    """Experiment with the GPT4All model."""
    ltmgpt.llm.streaming = True
    system_template = 'A chat between a curious user and an artificial intelligence assistant.'

    with ltmgpt.llm.gpt.chat_session(system_template):
        while True:
            user_text_request = input("You > ")

            bot_text_response = ltmgpt.response(user_text_request)
            
            if ltmgpt.llm.streaming:
                print(f"Bot <", end = ' ')
                for token in bot_text_response:
                    print(token, end = '')
                print()
            else:
                print(f"Bot < {bot_text_response}")

chat()