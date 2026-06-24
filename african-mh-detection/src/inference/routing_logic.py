"""
routing_logic.py
================
Logique de routage vers les ressources d'aide appropriées.
"""

from typing import Dict, List


RESOURCES = {
    "SOS_AMITIE_BURKINA": {
        "name": "SOS AMITIÉ Burkina Faso",
        "phone": "80 00 14 14",
        "hours": "24h/24, 7j/7",
        "cost": "Gratuit",
        "languages": ["français", "moore", "dioula"],
    },
    "CHU_YALGADO": {
        "name": "CHU Yalgado Ouédraogo — Service Psychiatrie",
        "address": "Ouagadougou, Burkina Faso",
        "hours": "Lun–Ven 7h30–12h30",
        "cost": "Consultation payante",
    },
    "GENERAL_SUPPORT": {
        "name": "Ressources générales",
        "description": "Parler à un proche de confiance, consulter un médecin",
    },
}


def route(
    categories: List[str],
    context_factors: List[str],
    signal_confidence: float,
) -> str:
    """
    Sélectionne la ressource la plus appropriée.

    Priorité:
    1. Idéation suicidaire → SOS AMITIÉ immédiat
    2. Dépression + solitude → SOS AMITIÉ
    3. Multiples catégories → Professionnel
    4. Signal faible → Support général

    Returns:
        Clé de ressource (str)
    """
    if "suicidal_ideation" in categories:
        return "SOS_AMITIE_BURKINA"

    if "depression" in categories and "loneliness" in context_factors:
        return "SOS_AMITIE_BURKINA"

    if len(categories) >= 2 or signal_confidence > 0.8:
        return "CHU_YALGADO"

    return "GENERAL_SUPPORT"


def get_resource_info(resource_key: str) -> Dict:
    """Retourne les informations sur une ressource."""
    return RESOURCES.get(resource_key, RESOURCES["GENERAL_SUPPORT"])
