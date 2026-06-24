"""
test_inference.py
=================
Tests pour l'API d'inférence DistressDetector.
"""

import pytest

from src.inference import DistressDetector


@pytest.fixture(scope="module")
def detector():
    """Fixture détecteur (chargé une fois pour tous les tests)."""
    return DistressDetector()  # Pas de modèle fine-tuné = poids aléatoires


class TestDistressDetector:
    def test_predict_returns_result(self, detector):
        """predict() doit retourner un DetectionResult."""
        result = detector.predict("Je ne vais pas bien du tout")
        assert result is not None
        assert hasattr(result, "signal")
        assert hasattr(result, "confidence")
        assert hasattr(result, "categories")

    def test_confidence_between_0_and_1(self, detector):
        """La confiance doit être entre 0 et 1."""
        result = detector.predict("Test")
        assert 0.0 <= result.confidence <= 1.0

    def test_signal_is_bool(self, detector):
        """signal doit être un booléen."""
        result = detector.predict("Test texte")
        assert isinstance(result.signal, bool)

    def test_categories_are_valid(self, detector):
        """Les catégories doivent être dans la liste valide."""
        from src.models.multitask_model import CATEGORIES
        result = detector.predict("Je souffre énormément")
        for cat in result.categories:
            assert cat in CATEGORIES

    def test_dialect_detection_french(self, detector):
        """Doit détecter le français."""
        result = detector.predict("Je suis triste aujourd'hui", dialect=None)
        assert result.dialect == "fr"

    def test_predict_batch(self, detector):
        """predict_batch doit retourner autant de résultats que d'entrées."""
        texts = ["Texte 1", "Texte 2", "Texte 3"]
        results = detector.predict_batch(texts)
        assert len(results) == len(texts)

    def test_routing_none_for_no_signal(self, detector):
        """Pas de routage si pas de signal (selon seuil)."""
        # Avec poids aléatoires, on teste juste la structure
        result = detector.predict("La journée s'est bien passée")
        if not result.signal:
            assert result.routing == "NONE"
            assert result.message == ""
