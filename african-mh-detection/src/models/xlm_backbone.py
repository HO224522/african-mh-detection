"""
xlm_backbone.py
===============
Wrapper autour de XLM-RoBERTa avec utilitaires de configuration.
"""

from transformers import AutoModel, AutoTokenizer, AutoConfig


def load_backbone(model_name: str = "xlm-roberta-base", from_checkpoint: str = None):
    """
    Charge le backbone XLM-RoBERTa.

    Args:
        model_name: Nom du modèle HuggingFace
        from_checkpoint: Chemin vers un checkpoint local (optionnel)

    Returns:
        (model, tokenizer, config)
    """
    source = from_checkpoint or model_name
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(source, config=config)
    return model, tokenizer, config
