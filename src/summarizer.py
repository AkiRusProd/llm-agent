import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from dotenv import dotenv_values

env = dotenv_values(".env")
os.environ['HUGGINGFACE_HUB_CACHE'] = env['HUGGINGFACE_HUB_CACHE']

# checkpoint = "t5-small"
# checkpoint = "google/mt5-small"
# checkpoint = "facebook/bart-large-cnn"
checkpoint = "sshleifer/distilbart-cnn-12-6"

# class Summarizer():
#     def __init__(self, model = checkpoint) -> None:
#         self.summarizer = pipeline("summarization", model = model)#, min_length = 30, max_length = 300

#     def summarize(self, text: str, min_length_ratio = 0.3, max_length_ratio = 1.):
#         if len(text) < 5:
#             return text

#         prompt = f"summarize: {text}"

#         return self.summarizer(prompt,  min_length = int(min_length_ratio * len(prompt.split(" "))), max_length = int(max_length_ratio * len(prompt.split(" "))))[0]['summary_text']

#     def __call__(self, text, min_length_ratio = 0.3, max_length_ratio = 1.):
#         return self.summarize(text, min_length_ratio, max_length_ratio)



# https://discuss.huggingface.co/t/summarization-on-long-documents/920/23
# https://www.width.ai/post/4-long-text-summarization-methods

class Summarizer():
    def __init__(self, model = checkpoint) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        # self.model = BartForConditionalGeneration.from_pretrained(model_name)#.to('cuda')
        # self.tokenizer = BartTokenizer.from_pretrained(model_name)

    def summarize(self, text: str, min_length = 30, max_length = 100):
        """Fixed-size chunking"""
        inputs_no_trunc = self.tokenizer(text, max_length=None, return_tensors='pt', truncation=False)
        if len(inputs_no_trunc['input_ids'][0]) < 30:
            return text

        # min_length = min_length_ratio * len(inputs)
        # max_length = max_length_ratio * len(inputs)
        
        inputs_batch_lst = []
        chunk_start = 0
        chunk_end = self.tokenizer.model_max_length  # == 1024 for Bart
        while chunk_start <= len(inputs_no_trunc['input_ids'][0]):
            inputs_batch = inputs_no_trunc['input_ids'][0][chunk_start:chunk_end]  # get batch of n tokens
            inputs_batch = torch.unsqueeze(inputs_batch, 0)
            inputs_batch_lst.append(inputs_batch)
            chunk_start += self.tokenizer.model_max_length  # == 1024 for Bart
            chunk_end += self.tokenizer.model_max_length  # == 1024 for Bart
        summary_ids_lst = [self.model.generate(inputs.to(self.device), num_beams=4, min_length=min_length, max_length=max_length, early_stopping=True) for inputs in inputs_batch_lst]

        summary_batch_lst = []
        for summary_id in summary_ids_lst:
            summary_batch = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            summary_batch_lst.append(summary_batch[0])
        summary_all = '\n'.join(summary_batch_lst)

        return summary_all

    def __call__(self, text, min_length = 30, max_length = 100):
        return self.summarize(text, min_length, max_length)