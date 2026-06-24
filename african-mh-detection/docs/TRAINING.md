# 🏋️ Guide d'Entraînement

## Vue d'ensemble

L'entraînement se fait en **3 phases progressives** pour éviter le catastrophic forgetting du backbone XLM-RoBERTa.

---

## Prérequis

```bash
# GPU recommandé (NVIDIA, 8GB+ VRAM)
# CPU possible mais 10x plus lent

pip install -r requirements.txt

# Vérifier GPU disponible
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Préparation des données

### Format requis

```csv
text,task1_signal,task2_category,task3_context,dialect
"Je souffre...",1,depression|isolation,loneliness,french
"La vie est belle",0,,,french
```

### Taille minimale recommandée

| Dataset | Minimum | Recommandé |
|---|---|---|
| Total | 500 textes | 2500 textes |
| Positifs (signal=1) | 200 | 1000 |
| Négatifs (signal=0) | 300 | 1500 |
| Par dialecte | 50 | 300 |

### Split automatique

Le script `train_multitask_model.py` crée automatiquement :
- Train : 70%
- Validation : 15%
- Test : 15%

---

## Phase 1 — Têtes uniquement (3 epochs)

Le backbone XLM-RoBERTa est **gelé**. Seules les couches custom sont entraînées.

**Pourquoi** : Initialiser les têtes de prédiction avant de perturber le backbone.

```bash
# Automatique via le script principal
python scripts/train_multitask_model.py \
    --corpus data/corpus/annotated_texts.csv \
    --output data/models/ \
    --epochs 30
```

**Métriques attendues en fin de Phase 1** :
- Train Loss : ~0.8
- Val F2 : ~0.50–0.60

---

## Phase 2 — Dégel progressif (6 epochs)

Les **4 dernières couches** du backbone sont dégelées.

**Pourquoi** : Les dernières couches de transformers encodent des représentations plus spécifiques aux tâches, donc plus faciles à adapter.

**Learning rate** : 5e-5 (plus bas qu'en Phase 1)

**Métriques attendues en fin de Phase 2** :
- Train Loss : ~0.5
- Val F2 : ~0.70–0.75

---

## Phase 3 — Fine-tuning complet (21 epochs)

Tout le modèle est dégelé avec un learning rate très faible et un scheduler linéaire.

**Learning rate** : 2e-5 avec warmup
**Scheduler** : Linear decay après warmup (200 steps)
**Gradient clipping** : 1.0

**Métriques cibles** :
- Val F2 : ≥ 0.85
- Val Recall : ≥ 90%
- Biais max entre dialectes : < 5%

---

## Calibration Platt Scaling

Après entraînement, calibrer les probabilités pour des seuils fiables :

```bash
python scripts/calibrate_thresholds.py \
    --model data/models/best_model.pt \
    --corpus data/corpus/annotated_texts.csv \
    --output data/models/calibration_params.pkl
```

---

## Audit de Biais

**Obligatoire** avant déploiement :

```bash
python scripts/audit_bias.py \
    --corpus data/corpus/annotated_texts.csv \
    --model data/models/best_model.pt \
    --output reports/bias_audit_initial.json
```

Le système alerte si l'écart de F2 entre dialectes dépasse 5%.

---

## Évaluation Complète

```bash
python scripts/evaluate_model.py \
    --model data/models/best_model.pt \
    --corpus data/corpus/annotated_texts.csv
```

Sortie attendue :
```
=== EVALUATION COMPLÈTE ===
Task 1 - Signal Detection:
  F2-Score:   0.873
  Recall:     0.912
  Precision:  0.801
  
Task 2 - Categories:
  mAP:        0.821

Task 3 - Context:
  mAP:        0.756

Bias Audit:
  Max gap:    0.032 (< 0.05 ✅)
  
Latency (CPU): 1.82s / sample
Model size:    284MB (→ 71MB quantized)
```

---

## Quantization pour Mobile

```bash
python scripts/quantize_model.py \
    --model data/models/best_model.pt \
    --output data/models/model_int8.onnx
```

---

## Troubleshooting

### CUDA out of memory
```bash
# Réduire batch size
python scripts/train_multitask_model.py --batch-size 8
```

### Loss ne diminue pas
- Vérifier que le corpus est équilibré (positifs/négatifs)
- Réduire le learning rate : `--lr 1e-5`
- Vérifier format CSV (colonnes bien nommées)

### F2-Score plafonné à 0.6
- Augmenter les données (minimum 500 textes positifs)
- Vérifier la qualité des annotations (Kappa ≥ 0.75)
