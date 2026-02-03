import torch
from transformers import DistilBertTokenizerFast, DistilBertModel


class DistilBertLanguageEncoder:
    """
    Thin wrapper around DistilBERT to produce sentence-level embeddings
    (CLS token) for language conditioning.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts):
        """
        texts: list of strings, length B
        return: Tensor (B, hidden_dim) on self.device
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        ).to(self.device)

        outputs = self.model(**inputs).last_hidden_state  # (B, L, D)
        cls = outputs[:, 0]  # (B, D)
        return cls

