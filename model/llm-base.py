from transformers import AutoTokenizer , AutoModelForCausalLM

class LLMBase():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("geobrain-ai/geogalactica")
        self.model = AutoModelForCausalLM.from_pretrained("geobrain-ai/geogalactica")
    def get_model(self):
        return self.model
    def get_tokenizer(self):
        return self.tokenizer
