# embedding/schema_encoder.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class SchemaEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', device=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, schema_strings, max_len=32):
        if len(schema_strings) == 0:
            return torch.empty(0, self.model.config.hidden_size).to(self.device)
        inputs = self.tokenizer(schema_strings, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS
