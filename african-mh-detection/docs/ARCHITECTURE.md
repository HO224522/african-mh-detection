# 🏗️ Architecture Technique

## Backbone: XLM-RoBERTa

XLM-RoBERTa (`xlm-roberta-base`) est choisi pour :

- **Multilingue natif** : pré-entraîné sur 100 langues dont le français
- **Robustesse** : Meilleur que mBERT sur les langues à faibles ressources
- **Taille raisonnable** : 278M paramètres (base), déployable
- **Code-switching** : Gère naturellement les mélanges de langues

## Architecture Multi-Tâches

```
Input: "Je ne vois plus d'avenir"
         │
         ▼
[XLM-RoBERTa Tokenizer]
         │
         ▼
[Token Embeddings] → 256 tokens max
         │
         ▼
[XLM-RoBERTa Encoder] (12 layers × 768 hidden)
         │
    [CLS] token → représentation globale du texte
         │
         ▼
[Shared Dense Layers]
  Linear(768 → 256) + ReLU + Dropout(0.1)
  Linear(256 → 256) + ReLU + Dropout(0.1)
         │
    ┌────┴────┬─────────┐
    ▼         ▼         ▼
[Head 1]  [Head 2]  [Head 3]
Linear    Linear    Linear
(256→2)  (256→5)   (256→7)
Signal    Catég.    Contexte
```

## Loss Combinée

```
L_total = 0.50 × L_signal + 0.30 × L_category + 0.20 × L_context
```

- **L_signal** : CrossEntropyLoss (binaire, avec class weights)
- **L_category** : BCEWithLogitsLoss (multi-label)
- **L_context** : BCEWithLogitsLoss (multi-label)

## Calibration Platt Scaling

Après entraînement, les probabilités brutes sont calibrées via Platt Scaling pour améliorer la fiabilité des seuils :

```python
from sklearn.calibration import CalibratedClassifierCV
# Wrapper appliqué aux logits Task 1
```

## Seuils Asymétriques (F2-optimisés)

Les seuils de décision sont optimisés pour maximiser le F2-score (rappel 2× plus important que précision) :

```
Threshold_signal = argmax F2(threshold) sur le val set
```

Valeur typique : **0.35** (vs 0.5 par défaut)

→ Rappel ≥ 90%, Précision ≥ 75%

## Inférence On-Device

Pour préserver la vie privée, l'inférence peut se faire directement sur l'appareil mobile :

1. Modèle quantisé int8 (ONNX) : ~70MB
2. Runtime ONNX sur Android/iOS
3. Le texte ne quitte jamais l'appareil
4. Seul le signal (binaire) est envoyé au serveur pour le routage
