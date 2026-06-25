#  Guide d'Annotation du Corpus

## Vue d'ensemble

Ce guide explique comment annoter les textes pour les 3 tâches du système de détection.

**3 annotateurs indépendants** par texte — accord inter-annotateurs requis (Kappa ≥ 0.75).

---

## Format CSV

```
text, task1_signal, task2_category, task3_context, dialect, annotator_notes
```

---

## Task 1 — Détection Signal (BINAIRE)

| Valeur | Signification |
|---|---|
| `1` | Signal de détresse présent |
| `0` | Pas de signal (neutre ou positif) |

**Critères signal = 1** :
- Expression explicite de souffrance, tristesse intense, désespoir
- Mention d'idées de mort ou de disparition
- Isolement social exprimé comme douloureux
- Épuisement extrême empêchant le fonctionnement
- Sentiment de ne plus avoir de valeur ou d'utilité

** Attention** : La politesse culturelle peut masquer la détresse. "Ça va, je me débrouille" peut cacher une souffrance en contexte burkinabè.

---

## Task 2 — Catégorisation (MULTI-LABEL)

Valeurs possibles (séparer par `|`) :

| Code | Description | Exemple |
|---|---|---|
| `depression` | Tristesse chronique, anhédonie | "Je ne ressens plus rien" |
| `anxiety` | Inquiétude excessive, rumination | "Je ne dors plus, j'ai peur" |
| `suicidal_ideation` | Pensées de mort ou disparition | "Je pense à partir pour toujours" |
| `isolation` | Retrait social douloureux | "Personne ne m'écoute plus" |
| `burnout` | Épuisement profond | "Je n'en peux plus, je suis à bout" |

**Exemple** : `depression|isolation`

---

## Task 3 — Contexte (MULTI-LABEL)

| Code | Description |
|---|---|
| `grief` | Deuil, perte d'un proche |
| `unemployment` | Chômage, perte d'emploi |
| `violence` | Violence subie (familiale, sociale) |
| `illness` | Maladie (soi ou proche) |
| `loneliness` | Solitude structurelle |
| `family_conflict` | Conflits familiaux |
| `financial_stress` | Difficultés financières |

---

## Dialecte

| Code | Description |
|---|---|
| `french` | Français standard ou local |
| `moore` | Moore (langue principale Burkina) |
| `dioula` | Dioula/Bambara |
| `mix` | Code-switching (mélange) |

---

## Cas ambigus

- En cas de doute : privilégier signal = 1 (mieux vaut un faux positif qu'un faux négatif)
- Expressions idiomatiques : consulter un locuteur natif
- Notes annotateur : expliquer les cas difficiles dans la colonne `annotator_notes`

---

## Processus

1. Lire le texte 2 fois
2. Annoter Task 1 d'abord
3. Si Task 1 = 1, annoter Task 2 et Task 3
4. Indiquer le dialecte
5. Ajouter note si cas difficile
6. Ne pas consulter les autres annotateurs avant de finir

---

*Voir également : ETHICS.md pour le cadre éthique de l'annotation*
