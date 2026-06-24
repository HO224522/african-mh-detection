"""
tokenizer.py
============
Wrapper de tokenisation pour XLM-RoBERTa avec support multilingue.
"""

from typing import Dict, List, Union
import torch
from transformers import AutoTokenizer


class MultilingualTokenizer:
    """
    Tokenizer multilingue pour French/Moore/Dioula.
    Wrapper autour de XLM-RoBERTa tokenizer.
    """

    def __init__(self, model_name: str = "xlm-roberta-base", max_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def encode(self, text: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """Encode un ou plusieurs textes."""
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **kwargs,
        )

    def decode(self, token_ids: torch.Tensor) -> str:
        """Décode des token IDs en texte."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
