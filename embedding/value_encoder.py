# embedding/value_encoder.py
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

class ValueEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', device=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, texts, max_len=32):
        if len(texts) == 0:
            return torch.empty(0, self.model.config.hidden_size).to(self.device)
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token