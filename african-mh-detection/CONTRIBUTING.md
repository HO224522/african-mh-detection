# 🤝 Guide de Contribution

Merci de votre intérêt pour ce projet ! Voici comment contribuer.

---

## Types de contributions acceptées

| Type | Description |
|---|---|
| 🐛 **Bug fixes** | Corriger des erreurs dans le code |
| ✨ **Features** | Nouvelles fonctionnalités |
| 📝 **Documentation** | Améliorer les docs, exemples |
| 🌍 **Traductions** | Moore, Dioula, Bambara... |
| 📊 **Annotation** | Annoter des textes pour le corpus |
| 🧪 **Tests** | Ajouter ou améliorer les tests |

---

## Setup développement

```bash
# 1. Fork le repo sur GitHub

# 2. Cloner votre fork
git clone https://github.com/VOTRE-USERNAME/african-mh-detection.git
cd african-mh-detection

# 3. Créer environnement virtuel
python -m venv venv
source venv/bin/activate

# 4. Installer dépendances + pre-commit
pip install -r requirements.txt
pre-commit install

# 5. Créer une branche
git checkout -b feature/ma-fonctionnalite

# 6. Coder + tester
pytest tests/ -v

# 7. Push & ouvrir PR
git push origin feature/ma-fonctionnalite
```

---

## Standards de code

- **Style** : PEP 8, formaté avec `black`
- **Types** : Type hints complets (vérifiés avec `mypy`)
- **Docstrings** : Google style pour toutes les fonctions publiques
- **Tests** : Tout nouveau code doit avoir des tests (pytest)
- **Lint** : Aucune erreur `flake8`

```bash
# Avant de push, vérifier :
black src/ tests/ scripts/
flake8 src/ --max-line-length=100
mypy src/
pytest tests/ -v --cov=src
```

---

## Processus Pull Request

1. Décrire clairement ce que fait la PR
2. Référencer l'issue GitHub si applicable (`Closes #42`)
3. Tous les tests doivent passer
4. Attendre la review avant merge

---

## Annotation du Corpus

Si vous souhaitez contribuer à l'annotation de textes :

1. Lire `data/corpus/ANNOTATION_GUIDE.md`
2. Contacter l'équipe via GitHub Issues
3. Recevoir accès au formulaire d'annotation
4. Annoter 50 textes minimum (3 tâches par texte)
5. Calcul accord inter-annotateurs (Kappa ≥ 0.75 requis)

---

## Code de conduite

- Respectueux et inclusif
- Pas de données personnelles réelles dans les PR
- Sensibilité aux enjeux de santé mentale
- Voir [ETHICS.md](docs/ETHICS.md)

---

## Questions ?

Ouvrir une [GitHub Issue](https://github.com/HO224522/african-mh-detection/issues) !
