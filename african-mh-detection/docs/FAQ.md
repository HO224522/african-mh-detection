# ❓ FAQ

## Technique

**Q: Quel GPU est nécessaire ?**  
R: Aucun en inférence. Le modèle tourne sur CPU en < 2s. L'entraînement bénéficie d'un GPU (NVIDIA T4 minimum).

**Q: Le modèle fonctionne-t-il hors ligne ?**  
R: Oui, c'est le principe de l'inférence on-device. Une fois téléchargé, le modèle n'a besoin d'aucune connexion.

**Q: Comment ajouter une nouvelle langue ?**  
R: XLM-RoBERTa supporte 100 langues nativement. Il suffit d'annoter des données dans la nouvelle langue et de fine-tuner.

## Éthique

**Q: Le système peut-il se tromper ?**  
R: Oui. C'est pourquoi nous optimisons le rappel (recall ≥ 90%) plutôt que la précision, et nous ne posons jamais de diagnostic.

**Q: Les données des utilisateurs sont-elles partagées ?**  
R: Non. L'inférence est on-device et aucun contenu n'est transmis. Voir [ETHICS.md](ETHICS.md).

**Q: Comment signaler un biais ou une erreur ?**  
R: Ouvrir une [GitHub Issue](https://github.com/HO224522/african-mh-detection/issues) ou contacter directement l'équipe.

## Compétition

**Q: Ce projet est-il open source ?**  
R: Oui, licence MIT. Librement adaptable par les ONG et gouvernements.

**Q: Comment contribuer ?**  
R: Voir [CONTRIBUTING.md](../CONTRIBUTING.md).
