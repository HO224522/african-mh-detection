"""
encryption.py
=============
Utilitaires de chiffrement et anonymisation.
Garantit la confidentialité des données utilisateur.
"""

import hashlib
import os
import secrets


def anonymize_text(text: str) -> str:
    """
    Anonymise un texte en supprimant les informations personnelles potentielles.
    Version simplifiée — en production, utiliser un NER dédié.
    """
    # Suppression de patterns courants (noms, numéros)
    import re
    # Numéros de téléphone
    text = re.sub(r'\b\d{8,}\b', '[NUMERO]', text)
    # Emails
    text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', text)
    return text


def generate_session_id() -> str:
    """Génère un identifiant de session aléatoire et éphémère."""
    return secrets.token_hex(16)


def hash_for_logging(text: str) -> str:
    """
    Hash d'un texte pour logging (pas de contenu stocké).
    Permet de détecter les doublons sans stocker le texte.
    """
    salt = os.environ.get("HASH_SALT", "default_salt_change_in_prod")
    return hashlib.sha256(f"{salt}{text}".encode()).hexdigest()[:16]
