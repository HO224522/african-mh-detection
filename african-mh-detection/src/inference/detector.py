"""
detector.py
===========
API principale d'inférence pour la détection de détresse psychologique.

Usage:
    from src.inference import DistressDetector

    detector = DistressDetector()
    result = detector.predict("Je ne vois plus d'avenir")
    print(result)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.models.multitask_model import (
    BACKBONE_MODEL,
    CATEGORIES,
    CONTEXT_FACTORS,
    MultiTaskDistressDetector,
    TrainingConfig,
)

logger = logging.getLogger(__name__)

# ─── Seuils par défaut (F2-optimisés) ───────────────────────────────────────

DEFAULT_THRESHOLDS = {
    "signal": 0.35,      # Bas = haute sensibilité (Recall ≥ 90%)
    "category": 0.40,
    "context": 0.35,
}

# ─── Messages de routage multilingues ───────────────────────────────────────

ROUTING_MESSAGES = {
    "SOS_AMITIE_BURKINA": {
        "fr": "Nous sommes là pour vous. Contactez SOS AMITIÉ au 80 00 14 14 (gratuit, 24h/24).",
        "moore": "Tõnd bee ne yãmb yẽ. Bool SOS AMITIÉ: 80 00 14 14.",
        "dioula": "An be aw fe. Weele SOS AMITIÉ: 80 00 14 14.",
    },
    "PROFESSIONAL_HELP": {
        "fr": "Parler à un professionnel peut vraiment aider. Consultez le CHU Yalgado ou un médecin.",
        "moore": "Yɩl-tõndo soaba tõe sõng-y lame. Kẽng CHU Yalgado.",
        "dioula": "Kuma dɔgɔtɔrɔ la, o bena kɛnɛ. Taa CHU Yalgado.",
    },
    "GENERAL_SUPPORT": {
        "fr": "Vous n'êtes pas seul(e). Des ressources sont disponibles pour vous soutenir.",
        "moore": "Fo ka yɩ yembre. Sõng-rãmba bee f yĩnga.",
        "dioula": "Aw tɛ kelen ye. Dɛmɛbagaw be aw ye.",
    },
}


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Résultat complet de détection."""

    signal: bool
    confidence: float
    categories: List[str] = field(default_factory=list)
    context_factors: List[str] = field(default_factory=list)
    routing: str = "GENERAL_SUPPORT"
    message: str = ""
    dialect: str = "fr"
    raw_probs: Optional[Dict] = None


# ─── Détecteur Principal ─────────────────────────────────────────────────────

class DistressDetector:
    """
    API principale de détection de détresse psychologique.

    Fonctionnalités:
    - Inférence multi-tâches (signal, catégorie, contexte)
    - Calibration Platt scaling
    - Seuils asymétriques (F2-optimisés, rappel prioritaire)
    - Messages personnalisés FR/Moore/Dioula
    - Détection automatique du dialecte

    Args:
        model_path: Chemin vers le modèle fine-tuné (.pt)
        thresholds_path: Chemin vers les seuils JSON optimisés
        device: "cuda", "cpu", ou "auto"
        dialect: "fr", "moore", "dioula" (détection auto si None)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        thresholds_path: Optional[str] = None,
        device: str = "auto",
        dialect: str = "fr",
    ):
        self.dialect = dialect
        self.thresholds = DEFAULT_THRESHOLDS.copy()

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Device: {self.device}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL)

        # Modèle
        self.model = MultiTaskDistressDetector(TrainingConfig())
        if model_path and os.path.exists(model_path):
            logger.info(f"Chargement modèle: {model_path}")
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
        else:
            logger.warning("Aucun modèle fine-tuné trouvé — utilisation poids initiaux")

        self.model.to(self.device)
        self.model.eval()

        # Seuils optimisés
        if thresholds_path and os.path.exists(thresholds_path):
            with open(thresholds_path) as f:
                self.thresholds = json.load(f)
            logger.info(f"Seuils chargés: {self.thresholds}")

    def detect_dialect(self, text: str) -> str:
        """
        Détection heuristique du dialecte.
        À remplacer par un classifieur dédié en production.
        """
        moore_markers = ["yaa", "tõe", "rãmba", "yɩl", "sõng", "wã"]
        dioula_markers = ["be", "taa", "ka", "ye", "bena", "la", "ani"]

        text_lower = text.lower()
        moore_score = sum(1 for m in moore_markers if m in text_lower)
        dioula_score = sum(1 for m in dioula_markers if m in text_lower)

        if moore_score > dioula_score and moore_score > 0:
            return "moore"
        elif dioula_score > moore_score and dioula_score > 0:
            return "dioula"
        return "fr"

    def _determine_routing(
        self, categories: List[str], context_factors: List[str]
    ) -> str:
        """Logique de routage basée sur catégories et contexte."""
        if "suicidal_ideation" in categories:
            return "SOS_AMITIE_BURKINA"
        if "depression" in categories and "loneliness" in context_factors:
            return "SOS_AMITIE_BURKINA"
        if len(categories) >= 2:
            return "PROFESSIONAL_HELP"
        return "GENERAL_SUPPORT"

    def predict(self, text: str, dialect: Optional[str] = None) -> DetectionResult:
        """
        Prédiction pour un texte unique.

        Args:
            text: Texte à analyser
            dialect: Dialecte (auto-détecté si None)

        Returns:
            DetectionResult avec signal, catégories, contexte, message
        """
        detected_dialect = dialect or self.detect_dialect(text)

        # Tokenisation
        encoded = self.tokenizer(
            text,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Inférence
        with torch.no_grad():
            output = self.model(**encoded)

        # Probabilités
        signal_prob = F.softmax(output.signal_logits, dim=-1)[0, 1].item()
        category_probs = torch.sigmoid(output.category_logits)[0].tolist()
        context_probs = torch.sigmoid(output.context_logits)[0].tolist()

        # Application des seuils
        signal = signal_prob >= self.thresholds["signal"]
        categories = [
            CATEGORIES[i]
            for i, p in enumerate(category_probs)
            if p >= self.thresholds["category"]
        ]
        context_factors = [
            CONTEXT_FACTORS[i]
            for i, p in enumerate(context_probs)
            if p >= self.thresholds["context"]
        ]

        # Routage
        routing = self._determine_routing(categories, context_factors) if signal else "NONE"
        message = ""
        if signal and routing != "NONE":
            message = ROUTING_MESSAGES[routing].get(detected_dialect, ROUTING_MESSAGES[routing]["fr"])

        return DetectionResult(
            signal=signal,
            confidence=round(signal_prob, 4),
            categories=categories,
            context_factors=context_factors,
            routing=routing,
            message=message,
            dialect=detected_dialect,
            raw_probs={
                "signal": round(signal_prob, 4),
                "categories": {c: round(p, 4) for c, p in zip(CATEGORIES, category_probs)},
                "context": {f: round(p, 4) for f, p in zip(CONTEXT_FACTORS, context_probs)},
            },
        )

    def predict_batch(self, texts: List[str]) -> List[DetectionResult]:
        """Prédiction par lots pour efficacité."""
        return [self.predict(t) for t in texts]


# ─── FastAPI App (optionnel) ─────────────────────────────────────────────────

try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(
        title="African MH Detection API",
        description="Détection de détresse psychologique en Afrique de l'Ouest",
        version="0.1.0",
    )

    detector = DistressDetector()

    class PredictRequest(BaseModel):
        text: str
        dialect: Optional[str] = None

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "african-mh-detection", "version": "0.1.0"}

    @app.post("/predict")
    def predict(request: PredictRequest):
        result = detector.predict(request.text, request.dialect)
        return {
            "signal": result.signal,
            "confidence": result.confidence,
            "categories": result.categories,
            "context_factors": result.context_factors,
            "routing": result.routing,
            "message": result.message,
            "dialect": result.dialect,
        }

except ImportError:
    pass  # FastAPI optionnel


# ─── Exemple d'utilisation ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    detector = DistressDetector()

    test_texts = [
        ("Je ne vois plus d'avenir, tout s'effondre", "fr"),
        ("La vie est belle, je suis heureux aujourd'hui!", "fr"),
        ("M bi kɛ dɔ, n ma se ka fɛ", "moore"),
        ("Tout va bien, merci", "fr"),
    ]

    for text, dialect in test_texts:
        result = detector.predict(text, dialect)
        print(f"\nTexte: {text[:50]}...")
        print(f"Signal: {result.signal} (conf: {result.confidence:.2%})")
        print(f"Catégories: {result.categories}")
        print(f"Contexte: {result.context_factors}")
        if result.message:
            print(f"Message: {result.message}")
