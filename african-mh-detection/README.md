# 🧠 African Mental Health Signal Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CI/CD](https://github.com/HO224522/african-mh-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/HO224522/african-mh-detection/actions)

> **Système d'alerte précoce pour la détection de détresse psychologique en Afrique de l'Ouest**  
> Soumission pour la compétition **Presidential African Youth in AI 2026** (Égypte)  
> Catégories : *Community Impact & Good Governance* + *Ethical AI*

---

## 🎯 Vue d'ensemble

Ce projet développe un système NLP multi-tâches capable de détecter des signaux de détresse psychologique dans les langues d'Afrique de l'Ouest (français, Moore, Dioula, code-switching), tout en respectant la vie privée des utilisateurs et en appliquant un cadre éthique rigoureux.

**Problème** : La santé mentale reste un tabou en Afrique de l'Ouest. Les ressources sont rares, peu accessibles, et les signaux de détresse passent souvent inaperçus.

**Solution** : Un système d'IA qui :
- Détecte discrètement les signaux de détresse dans les messages texte
- Catégorise le type de détresse (dépression, anxiété, isolement, etc.)
- Analyse le contexte (facteurs aggravants, ressources disponibles)
- Route intelligemment vers des ressources appropriées (ONG, services)
- Préserve l'anonymat total de l'utilisateur

---

## ✨ Fonctionnalités Clés

| Fonctionnalité | Description |
|---|---|
| 🤖 **Multi-task Learning** | 3 sorties simultanées : signal, catégorie, contexte |
| 🌍 **Multilingue** | Français, Moore, Dioula, code-switching |
| 🔒 **Privacy-First** | Inférence on-device, chiffrement bout-en-bout |
| ⚖️ **Fairness Audit** | Audit mensuel des biais par dialecte |
| 🏥 **NGO Partnership** | Intégration avec SOS AMITIÉ, Université Ki-Zerbo |
| 📱 **Mobile-Ready** | Modèle quantisé <100MB pour Android/iOS |

---

## 🏗️ Architecture

```
Input Text (FR/Moore/Dioula)
         │
         ▼
┌─────────────────────┐
│   XLM-RoBERTa       │  ← Backbone multilingue
│   (Fine-tuned)      │
└─────────┬───────────┘
          │
    ┌─────┴──────┐
    │  Shared    │
    │  Layers    │
    └──┬──┬──┬───┘
       │  │  │
  ┌────┘  │  └────┐
  ▼       ▼       ▼
Task1   Task2   Task3
Signal  Categ.  Context
(bin.)  (multi) (multi)
```

**Architecture Multi-Tâches** :
- **Task 1** : Détection signal (binaire : détresse / pas détresse)
- **Task 2** : Catégorisation (dépression, anxiété, idées noires, isolement, épuisement)
- **Task 3** : Contexte (facteurs : deuil, chômage, violence, maladie, solitude)

---

## 🚀 Démarrage Rapide

### Installation

```bash
git clone https://github.com/HO224522/african-mh-detection.git
cd african-mh-detection
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Utilisation

```python
from src.inference import DistressDetector

detector = DistressDetector()
result = detector.predict("Je ne vois plus d'avenir, tout s'effondre")
print(result)
# {
#   "signal": True,
#   "confidence": 0.87,
#   "category": "depression",
#   "context": ["isolation", "hopelessness"],
#   "routing": "SOS_AMITIE_BURKINA",
#   "message": "Nous sommes là pour vous..."
# }
```

### Avec Docker

```bash
docker build -t african-mh-api:latest .
docker run -p 8000:8000 african-mh-api:latest
# API disponible sur http://localhost:8000/docs
```

---

## 📊 Performance Cible

| Métrique | Cible | Justification |
|---|---|---|
| **F2-Score** | ≥ 0.85 | Optimisé pour le rappel |
| **Recall** | ≥ 90% | Faux négatif = danger |
| **Précision** | ≥ 75% | Trade-off acceptable |
| **Biais max** | < 5% gap | Fairness par dialecte |
| **Latence CPU** | < 2s | Utilisable sans GPU |
| **Taille modèle** | < 100MB | Déployable sur mobile |

---

## 📁 Structure du Projet

```
african-mh-detection/
├── src/
│   ├── models/
│   │   ├── multitask_model.py    # Architecture NLP multi-tâches
│   │   ├── xlm_backbone.py       # Backbone XLM-RoBERTa
│   │   └── loss_functions.py     # Loss pondérée multi-tâches
│   ├── inference/
│   │   ├── detector.py           # API principale d'inférence
│   │   └── routing_logic.py      # Routage vers ressources
│   └── utils/
│       ├── data_loader.py        # Chargement corpus
│       ├── tokenizer.py          # Tokenisation XLM
│       ├── bias_audit.py         # Audit de fairness
│       └── encryption.py        # Chiffrement & anonymisation
├── scripts/
│   ├── train_multitask_model.py  # Pipeline entraînement
│   ├── calibrate_thresholds.py   # Calibration Platt scaling
│   ├── audit_bias.py             # Audit biais mensuel
│   └── evaluate_model.py         # Évaluation complète
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_threshold_optimization.ipynb
│   └── 04_bias_audit.ipynb
├── docs/
│   ├── ARCHITECTURE.md
│   ├── TRAINING.md
│   ├── ETHICS.md
│   ├── DEPLOYMENT.md
│   └── FAQ.md
├── data/
│   └── corpus/
│       ├── sample_corpus.csv     # 50 textes annotés (exemple)
│       └── ANNOTATION_GUIDE.md
├── tests/
│   ├── test_model.py
│   ├── test_inference.py
│   └── test_bias_audit.py
├── Dockerfile
├── requirements.txt
├── CONTRIBUTING.md
└── LICENSE
```

---

## 🗺️ Feuille de Route

- **Phase 1** (Jan–Apr 2026) : Annotation corpus (1500 textes) + Partenariats ONG
- **Phase 2** (Apr–Mai 2026) : Entraînement modèle + Calibration seuils
- **Phase 3** (Mai–Jun 2026) : Pilote 500 utilisateurs (Burkina Faso)
- **Phase 4** (Oct 2026) : Présentation compétition Égypte

---

## 🤝 Partenaires

- **SOS AMITIÉ Burkina** — Ressources santé mentale & ligne d'écoute
- **Université Ki-Zerbo** — Validation scientifique & annotation corpus
- **Ministère de la Santé** — Pathway intégration systèmes publics (2028+)

---

## ⚖️ Éthique & Confidentialité

Ce système respecte des principes éthiques stricts :
- ❌ **Pas de diagnostic médical** — Orientation uniquement
- ❌ **Pas d'intervention forcée** — L'utilisateur garde le contrôle
- ✅ **Anonymat total** — Aucun identifiant stocké
- ✅ **Audit biais mensuel** — Équité entre dialectes
- ✅ **Transparent** — Code open-source, modèle explicable

Voir [ETHICS.md](docs/ETHICS.md) pour le cadre complet.

---

## 🤝 Contribuer

Les contributions sont les bienvenues ! Voir [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
# Développeurs
git clone https://github.com/HO224522/african-mh-detection.git
pip install -r requirements.txt
pytest tests/ -v
```

---

## 📄 Licence

MIT License — Voir [LICENSE](LICENSE). Libre pour adaptation par ONG et gouvernements.

---

## 📬 Contact

**Compétition** : [Presidential African Youth in AI 2026](https://ele-vate.co.za)  
**GitHub** : [HO224522/african-mh-detection](https://github.com/HO224522/african-mh-detection)

---

*Fait avec ❤️ pour l'Afrique de l'Ouest | Burkina Faso 🇧🇫*
