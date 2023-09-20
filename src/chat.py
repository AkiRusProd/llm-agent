from ltm_agent import ltm_agent



def chat():
    """Experiment with the GPT4All model."""
    ltm_agent.llm.streaming = True
    system_template = 'A chat between a curious user and an artificial intelligence assistant.'

    with ltm_agent.llm.gpt.chat_session(system_template):
        while True:
            user_text_request = input("You > ")

            bot_text_response = ltm_agent.response(user_text_request)
            
            if ltm_agent.llm.streaming:
                print(f"Bot <", end = ' ')
                for token in bot_text_response:
                    print(token, end = '')
                print()
            else:
                print(f"Bot < {bot_text_response}")

chat()