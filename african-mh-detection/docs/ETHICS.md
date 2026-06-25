# ⚖️ Cadre Éthique

## Principes Fondamentaux

Ce système est conçu autour de principes éthiques stricts, développés en concertation avec des professionnels de santé mentale et des représentants des communautés ciblées.

---

## ❌ Ce que le système NE fait PAS

### Pas de diagnostic médical
Le système **ne diagnostique jamais** une pathologie mentale. Il détecte des signaux et oriente vers des ressources. Seul un professionnel de santé qualifié peut poser un diagnostic.

### Pas d'intervention forcée
Aucun signalement automatique aux autorités ou à la famille sans consentement explicite de l'utilisateur. L'utilisateur garde le contrôle total.

### Pas de stockage d'identifiants
Aucune donnée permettant d'identifier l'utilisateur n'est conservée. Chaque session est indépendante.

### Pas de jugement
Le système n'évalue pas si la souffrance est "légitime". Tout signal est traité avec la même bienveillance.

---

## ✅ Ce que le système FAIT

### Détection discrète
Signal détecté silencieusement, sans alarme visible. L'utilisateur est orienté doucement vers des ressources.

### Anonymat structurel
- Inférence on-device : le texte ne quitte jamais l'appareil
- Pas de logs de contenu
- Services de classification et de routage découplés
- Données agrégées uniquement (statistiques anonymisées)

### Consentement éclairé
L'utilisateur est informé au premier lancement que l'application peut détecter des signaux de détresse et proposer des ressources. Il peut désactiver cette fonctionnalité.

### Audit de fairness
Audit mensuel pour vérifier que le système est équitable entre les dialectes (Moore, Dioula, Français). Aucun groupe ne doit être moins bien servi.

---

## 🔒 Sécurité des données

| Aspect | Mesure |
|---|---|
| **Transmission** | TLS 1.3 uniquement |
| **Stockage** | Aucun contenu des messages |
| **Logs** | Métriques agrégées uniquement |
| **Partage** | Jamais avec tiers sans consentement |
| **Durée conservation** | Aucune (session uniquement) |

---

## 🌍 Sensibilité Culturelle

### Tabou de la santé mentale
En Afrique de l'Ouest, la santé mentale reste souvent tabou. Le système est conçu pour :
- Ne pas stigmatiser l'utilisateur
- Proposer des ressources comme "aide disponible" plutôt que "vous avez un problème"
- Respecter les croyances locales tout en proposant un soutien professionnel

### Langues locales
Les messages de routage sont disponibles en Moore, Dioula et Français. Les communautés locales ont participé à la traduction pour assurer la justesse culturelle.

---

## 👥 Gouvernance

- **Comité éthique** : Représentants ONG + Université + Communautés
- **Révision semestrielle** : Audit complet du système
- **Signalement** : Mécanisme pour signaler des biais ou erreurs
- **Transparence** : Rapports publics annuels sur la performance et les biais

---

## 📞 Ressources d'urgence

Si le système détecte un signal critique (idéation suicidaire) :

1. Le message de routage inclut immédiatement le **numéro SOS AMITIÉ : 80 00 14 14 ( remarquons ici que ce numero est a titre indicatif )**
2. Disponible 24h/24, gratuit, confidentiel
3. Disponible en français et en langues locales

---

*Ce cadre éthique est révisé semestriellement. Dernière révision : Juin 2026*
