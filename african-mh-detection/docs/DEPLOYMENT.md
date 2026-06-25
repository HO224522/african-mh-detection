#  Guide de Déploiement

## Options de déploiement

### 1. Local (développement)

```bash
git clone https://github.com/HO224522/african-mh-detection.git
cd african-mh-detection
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.inference.detector:app --reload --port 8000
```

API disponible sur `http://localhost:8000/docs`

### 2. Docker (production)

```bash
docker build -t african-mh-api:latest .
docker run -p 8000:8000 --env-file .env african-mh-api:latest
```

### 3. Mobile (Android/iOS)

Le modèle est exporté en ONNX et quantisé INT8 pour tenir sous 100MB :

```bash
python scripts/evaluate_model.py --export-onnx --quantize
# Sortie: data/models/distress_detector_int8.onnx
```

Intégration avec ONNX Runtime Mobile (Android) ou Core ML (iOS).

## Variables d'environnement

| Variable | Valeur par défaut | Description |
|---|---|---|
| `MODEL_PATH` | `data/models/` | Chemin vers les fichiers modèle |
| `THRESHOLD_SIGNAL` | `0.45` | Seuil de détection (recall-optimized) |
| `LOG_LEVEL` | `WARNING` | Niveau de log (pas de contenu utilisateur) |
| `MAX_TEXT_LENGTH` | `512` | Longueur max du texte en entrée |

## Checklist avant déploiement

- [ ] Tests passés (`pytest tests/ -v`)
- [ ] Audit biais exécuté (`python scripts/audit_bias.py`)
- [ ] Seuils calibrés (`python scripts/calibrate_thresholds.py`)
- [ ] Modèle quantisé pour mobile
- [ ] Variables d'environnement configurées
- [ ] TLS activé (jamais HTTP en production)
