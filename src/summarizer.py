import os
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:\\Code\\Huggingface_cache\\' #MOVE TO ENV

from transformers import pipeline

# checkpoint = "t5-small"
# checkpoint = "google/mt5-small"
checkpoint = "sshleifer/distilbart-cnn-12-6"

class Summarizer():
    def __init__(self, model = checkpoint) -> None:
        self.summarizer = pipeline("summarization", model = model)#, min_length = 30, max_length = 300

    def summarize(self, text: str):
        prompt = f"summarize: {text}"

        return self.summarizer(prompt,  min_length = int(0.3 * len(prompt.split(" "))), max_length = int(1. * len(prompt.split(" "))))[0]['summary_text']

    def __call__(self, text):
        return self.summarize(text)