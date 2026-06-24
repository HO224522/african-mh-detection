#!/bin/bash
# ============================================================
# SCRIPT DE DÉPLOIEMENT GITHUB — african-mh-detection
# Presidential African Youth in AI 2026
# ============================================================
# Usage: bash setup_github.sh [GITHUB_TOKEN]
# ============================================================

set -euo pipefail

REPO_URL="https://github.com/HO224522/african-mh-detection.git"
GITHUB_USER="HO224522"
MAIN_BRANCH="main"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║   African Mental Health Detection — GitHub Setup     ║"
echo "║   Presidential African Youth in AI 2026 🌍           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Token ─────────────────────────────────────────────────
if [ -n "${1:-}" ]; then
    TOKEN="$1"
else
    echo -e "${YELLOW}Entrez votre GitHub Personal Access Token${NC}"
    echo "(Settings → Developer settings → Personal access tokens → Fine-grained)"
    echo -e "${YELLOW}Permissions requises : Contents: Read & Write${NC}"
    read -s -p "Token: " TOKEN
    echo ""
fi

[ -z "$TOKEN" ] && { echo -e "${RED}❌ Token requis${NC}"; exit 1; }

AUTH_URL="https://${TOKEN}@github.com/${GITHUB_USER}/african-mh-detection.git"

# ── Vérifier qu'on est dans le bon dossier ────────────────
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    echo -e "${RED}❌ Lancez ce script depuis le dossier african-mh-detection/${NC}"
    exit 1
fi

# ── Git init ──────────────────────────────────────────────
echo -e "\n${BLUE}📁 Initialisation Git...${NC}"
if [ ! -d ".git" ]; then
    git init
    echo -e "${GREEN}✅ git init${NC}"
else
    echo -e "${GREEN}✅ Dépôt Git existant${NC}"
fi

git config user.name "$GITHUB_USER"
git config user.email "${GITHUB_USER}@users.noreply.github.com"

# ── Remote ────────────────────────────────────────────────
echo -e "\n${BLUE}🔗 Configuration du remote...${NC}"
git remote remove origin 2>/dev/null || true
git remote add origin "$AUTH_URL"
echo -e "${GREEN}✅ Remote configuré${NC}"

# ── Branch ────────────────────────────────────────────────
git checkout -b "$MAIN_BRANCH" 2>/dev/null || git checkout "$MAIN_BRANCH" 2>/dev/null || true

# ── Récupérer le contenu distant existant ─────────────────
echo -e "\n${BLUE}🔄 Synchronisation avec GitHub...${NC}"
git fetch origin "$MAIN_BRANCH" 2>/dev/null || true

# ── Staging ───────────────────────────────────────────────
echo -e "\n${BLUE}📦 Ajout des fichiers...${NC}"
git add .
echo -e "${GREEN}✅ Fichiers stagés${NC}"

echo -e "\n${BLUE}📋 Fichiers inclus dans le commit:${NC}"
git status --short

# ── Commit ────────────────────────────────────────────────
echo -e "\n${BLUE}💾 Création du commit...${NC}"

COMMIT_MSG="feat: initial project — multi-task NLP distress detection system

## What's included

### Core ML System
- MultiTaskDistressDetector: XLM-RoBERTa + 3 simultaneous prediction heads
  - Task 1: Binary distress signal detection
  - Task 2: Category classification (depression, anxiety, suicidal ideation, isolation, burnout)
  - Task 3: Context factor detection (grief, unemployment, violence, illness, loneliness...)
- Asymmetric loss weighting: Signal(50%) + Category(30%) + Context(20%)
- Progressive unfreezing training strategy (3 phases)

### Inference & Routing
- DistressDetector API with Platt scaling calibration
- Intelligent routing to local resources (SOS AMITIÉ Burkina: 80 00 14 14)
- Asymmetric thresholds optimized for high recall (F2-score target ≥ 0.85)

### Ethics & Privacy
- On-device inference: text never leaves the device
- Zero user identifier storage
- Monthly fairness audit by dialect (Moore, Dioula, French)
- Full ethical framework in docs/ETHICS.md

### Infrastructure
- CI/CD: GitHub Actions (pytest × Python 3.9/3.10/3.11, flake8, mypy, black)
- Docker: multi-stage production build
- Sample corpus: 50 annotated texts in FR/Moore/Dioula

### Languages supported
- French (français)
- Moore (mooré)
- Dioula (jula)
- Code-switching between all three

Competition: Presidential African Youth in AI 2026 (Egypt)
Categories: Community Impact & Good Governance + Ethical AI
Author: OUEDRAOGO WENDYAM HASSANE — Burkina Faso 🇧🇫"

git commit -m "$COMMIT_MSG" 2>/dev/null || echo -e "${YELLOW}ℹ️  Rien de nouveau à commiter${NC}"

# ── Push ──────────────────────────────────────────────────
echo -e "\n${BLUE}🚀 Push vers GitHub...${NC}"
git push origin "$MAIN_BRANCH" --force
echo -e "${GREEN}✅ Push réussi${NC}"

# ── Résumé ────────────────────────────────────────────────
echo -e "\n${GREEN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║   ✅  DÉPÔT EN LIGNE !                               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "👉  ${BLUE}https://github.com/HO224522/african-mh-detection${NC}"
echo ""
echo -e "${YELLOW}Prochaines étapes recommandées :${NC}"
echo "  1. Ajouter les Topics GitHub: african-ai, mental-health, nlp, burkina-faso, xlm-roberta"
echo "  2. Activer GitHub Discussions (pour la communauté)"
echo "  3. Créer une Release: v0.1.0-alpha"
echo "  4. Ajouter un badge Codecov dans le README"
echo "  5. Épingler le repo sur votre profil GitHub"
echo ""
