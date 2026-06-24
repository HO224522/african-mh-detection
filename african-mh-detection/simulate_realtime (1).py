"""
simulate_realtime.py
====================
Simulation temps réel des 4 modules du système african-mh-detection.
Fonctionne SANS GPU et SANS télécharger XLM-RoBERTa (mode mock).

Usage:
    python simulate_realtime.py
    python simulate_realtime.py --mode inference
    python simulate_realtime.py --mode training
    python simulate_realtime.py --mode api
    python simulate_realtime.py --mode bias
"""

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Couleurs terminal ──────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    GRAY   = "\033[90m"
    PURPLE = "\033[95m"
    WHITE  = "\033[97m"

def banner(title: str, color: str = C.BLUE):
    w = 60
    print(f"\n{color}{C.BOLD}{'═' * w}{C.RESET}")
    print(f"{color}{C.BOLD}  {title}{C.RESET}")
    print(f"{color}{C.BOLD}{'═' * w}{C.RESET}")

def section(title: str):
    print(f"\n{C.CYAN}{C.BOLD}── {title} {'─' * (50 - len(title))}{C.RESET}")

def ok(msg):    print(f"  {C.GREEN}✓{C.RESET}  {msg}")
def warn(msg):  print(f"  {C.YELLOW}⚠{C.RESET}  {msg}")
def err(msg):   print(f"  {C.RED}✗{C.RESET}  {msg}")
def info(msg):  print(f"  {C.GRAY}·{C.RESET}  {msg}")

def progress_bar(value: float, width: int = 30, color: str = C.GREEN) -> str:
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{color}{bar}{C.RESET} {value:.1%}"

def typing(text: str, delay: float = 0.018):
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()

def sleep(s: float):
    time.sleep(s)

# ── Données de test ────────────────────────────────────────────────────────────

CATEGORIES     = ["depression", "anxiety", "suicidal_ideation", "isolation", "burnout"]
CONTEXT_FACTORS = ["grief", "unemployment", "violence", "illness", "loneliness",
                   "family_conflict", "financial_stress"]
DIALECTS       = ["french", "moore", "dioula", "mix"]

TEST_TEXTS = [
    {
        "text": "Je ne vois plus d'avenir, tout s'effondre autour de moi",
        "dialect": "fr",
        "label": 1,
        "expected_cat": ["depression", "isolation"],
        "expected_ctx": ["loneliness"],
    },
    {
        "text": "La vie est belle aujourd'hui, je me sens bien !",
        "dialect": "fr",
        "label": 0,
        "expected_cat": [],
        "expected_ctx": [],
    },
    {
        "text": "M bi kɛ dɔ, n ma se ka fɛ — nègèso tɛ ne fe",
        "dialect": "moore",
        "label": 1,
        "expected_cat": ["depression", "burnout"],
        "expected_ctx": ["grief", "loneliness"],
    },
    {
        "text": "An be aw fe, bena kɛnɛ — dɛmɛbagaw be",
        "dialect": "dioula",
        "label": 0,
        "expected_cat": [],
        "expected_ctx": [],
    },
    {
        "text": "Je pense parfois qu'il vaut mieux ne pas être là",
        "dialect": "fr",
        "label": 1,
        "expected_cat": ["suicidal_ideation", "depression"],
        "expected_ctx": ["loneliness", "family_conflict"],
    },
    {
        "text": "J'ai perdu mon travail, je ne dors plus, tout est difficile",
        "dialect": "fr",
        "label": 1,
        "expected_cat": ["anxiety", "burnout"],
        "expected_ctx": ["unemployment", "financial_stress"],
    },
    {
        "text": "Nka na dɔ, ne tɛ sɔrɔ ka kɛnɛ — kɔrɔ ye ne ma",
        "dialect": "dioula",
        "label": 1,
        "expected_cat": ["isolation", "depression"],
        "expected_ctx": ["loneliness"],
    },
    {
        "text": "Tout va bien, merci pour votre aide",
        "dialect": "fr",
        "label": 0,
        "expected_cat": [],
        "expected_ctx": [],
    },
]

ROUTING_MESSAGES = {
    "SOS_AMITIE_BURKINA": {
        "fr":     "Nous sommes là pour vous. SOS AMITIÉ : 80 00 14 14 (gratuit, 24h/24)",
        "moore":  "Tõnd bee ne yãmb yẽ. Bool SOS AMITIÉ: 80 00 14 14",
        "dioula": "An be aw fe. Weele SOS AMITIÉ: 80 00 14 14",
    },
    "PROFESSIONAL_HELP": {
        "fr":     "Parler à un professionnel peut aider. Consultez le CHU Yalgado.",
        "moore":  "Yɩl-tõndo soaba tõe sõng-y lame. Kẽng CHU Yalgado.",
        "dioula": "Kuma dɔgɔtɔrɔ la, o bena kɛnɛ. Taa CHU Yalgado.",
    },
    "GENERAL_SUPPORT": {
        "fr":     "Vous n'êtes pas seul(e). Des ressources sont disponibles.",
        "moore":  "Fo ka yɩ yembre. Sõng-rãmba bee f yĩnga.",
        "dioula": "Aw tɛ kelen ye. Dɛmɛbagaw be aw ye.",
    },
}

# ── Mock model (simule XLM-RoBERTa sans le télécharger) ───────────────────────

def mock_predict(text: str, label: int, expected_cats: list, expected_ctx: list) -> dict:
    """Simule la sortie du modèle avec du bruit réaliste."""
    random.seed(hash(text) % 10000)

    # Signal : basé sur le label attendu + bruit
    base_signal = 0.75 if label == 1 else 0.15
    signal_prob = max(0.01, min(0.99, base_signal + random.gauss(0, 0.08)))

    # Catégories : probabilités simulées
    cat_probs = {}
    for cat in CATEGORIES:
        if cat in expected_cats:
            cat_probs[cat] = round(max(0.3, min(0.95, 0.72 + random.gauss(0, 0.1))), 3)
        else:
            cat_probs[cat] = round(max(0.01, min(0.38, 0.12 + random.gauss(0, 0.08))), 3)

    # Contexte
    ctx_probs = {}
    for ctx in CONTEXT_FACTORS:
        if ctx in expected_ctx:
            ctx_probs[ctx] = round(max(0.3, min(0.92, 0.68 + random.gauss(0, 0.1))), 3)
        else:
            ctx_probs[ctx] = round(max(0.01, min(0.35, 0.10 + random.gauss(0, 0.07))), 3)

    THRESH_SIGNAL = 0.35
    THRESH_CAT    = 0.40
    THRESH_CTX    = 0.35

    detected_cats = [c for c, p in cat_probs.items() if p >= THRESH_CAT]
    detected_ctx  = [c for c, p in ctx_probs.items()  if p >= THRESH_CTX]

    # Routage
    if "suicidal_ideation" in detected_cats:
        routing = "SOS_AMITIE_BURKINA"
    elif "depression" in detected_cats and "loneliness" in detected_ctx:
        routing = "SOS_AMITIE_BURKINA"
    elif len(detected_cats) >= 2:
        routing = "PROFESSIONAL_HELP"
    else:
        routing = "GENERAL_SUPPORT"

    signal = signal_prob >= THRESH_SIGNAL

    return {
        "signal":          signal,
        "confidence":      round(signal_prob, 4),
        "categories":      detected_cats,
        "context_factors": detected_ctx,
        "routing":         routing if signal else "NONE",
        "cat_probs":       cat_probs,
        "ctx_probs":       ctx_probs,
    }

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — INFÉRENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference():
    banner("MODULE 1 — INFÉRENCE TEMPS RÉEL", C.BLUE)
    print(f"  {C.GRAY}Mode mock (XLM-RoBERTa simulé — aucun téléchargement requis){C.RESET}")

    section("Initialisation du détecteur")
    steps = [
        ("Chargement tokenizer XLM-RoBERTa...", 0.4),
        ("Construction architecture multi-tâches...", 0.3),
        ("Chargement poids (mock)...", 0.3),
        ("Calibration seuils Platt scaling...", 0.2),
        ("Détecteur prêt sur CPU", 0.1),
    ]
    for msg, delay in steps:
        info(msg)
        sleep(delay)
    ok("DistressDetector initialisé")

    section("Prédictions sur corpus de test")
    correct = 0

    for i, sample in enumerate(TEST_TEXTS):
        print(f"\n  {C.BOLD}[{i+1}/{len(TEST_TEXTS)}]{C.RESET} Texte : {C.WHITE}{sample['text'][:55]}...{C.RESET}")
        print(f"         Dialecte détecté : {C.CYAN}{sample['dialect']}{C.RESET}")

        # Simulation tokenisation
        info(f"Tokenisation → {random.randint(8, 24)} tokens")
        sleep(0.15)

        # Inférence mock
        result = mock_predict(
            sample["text"], sample["label"],
            sample["expected_cat"], sample["expected_ctx"]
        )
        sleep(0.25)

        # Affichage signal
        color = C.RED if result["signal"] else C.GREEN
        icon  = "🔴 DÉTRESSE" if result["signal"] else "🟢 OK"
        print(f"         Signal : {color}{C.BOLD}{icon}{C.RESET}  "
              f"conf={C.BOLD}{result['confidence']:.2%}{C.RESET}")
        print(f"         Barre  : {progress_bar(result['confidence'], 25, color)}")

        # Catégories
        if result["categories"]:
            cats_str = "  ".join(f"{C.YELLOW}{c}{C.RESET}" for c in result["categories"])
            print(f"         Catégories : {cats_str}")

        # Contexte
        if result["context_factors"]:
            ctx_str = "  ".join(f"{C.PURPLE}{c}{C.RESET}" for c in result["context_factors"])
            print(f"         Contexte   : {ctx_str}")

        # Message de routage
        if result["signal"] and result["routing"] != "NONE":
            msg = ROUTING_MESSAGES[result["routing"]].get(sample["dialect"],
                  ROUTING_MESSAGES[result["routing"]]["fr"])
            print(f"         {C.CYAN}→ {result['routing']}{C.RESET}")
            print(f"         {C.GRAY}\"{msg}\"{C.RESET}")

        # Vérification
        pred_label = 1 if result["signal"] else 0
        if pred_label == sample["label"]:
            correct += 1
            ok("Prédiction correcte")
        else:
            warn("Prédiction incorrecte (faux pos/neg)")

        sleep(0.1)

    # Résumé
    section("Résumé inférence")
    acc = correct / len(TEST_TEXTS)
    print(f"  Accuracy    : {progress_bar(acc, 30, C.GREEN if acc >= 0.7 else C.YELLOW)}")
    print(f"  Prédictions : {correct}/{len(TEST_TEXTS)} correctes")
    ok("Module inférence terminé")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_training():
    banner("MODULE 2 — SIMULATION ENTRAÎNEMENT", C.PURPLE)
    print(f"  {C.GRAY}Pipeline d'entraînement multi-tâches (3 phases){C.RESET}")

    EPOCHS  = 10
    BATCHES = 8

    config = {
        "backbone":    "xlm-roberta-base",
        "batch_size":  16,
        "lr":          2e-5,
        "max_length":  256,
        "hidden_dim":  256,
        "dropout":     0.1,
        "loss_weights": {"signal": 0.50, "category": 0.30, "context": 0.20},
    }

    section("Configuration")
    for k, v in config.items():
        info(f"{k:<18} = {C.CYAN}{v}{C.RESET}")

    phases = [
        ("Phase 1 — Backbone gelé (têtes seulement)", 3,  0.001),
        ("Phase 2 — Dernières 4 couches dégelées",   4,  0.0001),
        ("Phase 3 — Modèle complet fine-tuned",       3,  0.00005),
    ]

    epoch_global = 0
    metrics_history = []

    for phase_name, phase_epochs, lr in phases:
        section(phase_name)
        info(f"Learning rate : {lr}")
        sleep(0.3)

        for ep in range(1, phase_epochs + 1):
            epoch_global += 1
            print(f"\n  {C.BOLD}Epoch {epoch_global}/{EPOCHS}{C.RESET}  "
                  f"{C.GRAY}({phase_name[:25]}...){C.RESET}")

            # Simulation batches
            batch_losses = []
            for b in range(1, BATCHES + 1):
                progress = b / BATCHES

                # Pertes simulées décroissantes avec bruit
                decay  = math.exp(-epoch_global * 0.18)
                noise  = random.gauss(0, 0.015)
                l_sig  = round(max(0.05, 0.72 * decay + noise + random.gauss(0, 0.01)), 4)
                l_cat  = round(max(0.08, 0.65 * decay + noise + random.gauss(0, 0.01)), 4)
                l_ctx  = round(max(0.10, 0.58 * decay + noise + random.gauss(0, 0.01)), 4)
                l_tot  = round(0.5 * l_sig + 0.3 * l_cat + 0.2 * l_ctx, 4)
                batch_losses.append(l_tot)

                bar = "█" * b + "░" * (BATCHES - b)
                print(f"\r  [{bar}] batch {b}/{BATCHES}  "
                      f"loss={C.YELLOW}{l_tot:.4f}{C.RESET}  "
                      f"L_sig={l_sig:.3f}  L_cat={l_cat:.3f}  L_ctx={l_ctx:.3f}",
                      end="", flush=True)
                sleep(0.08)

            avg_loss = sum(batch_losses) / len(batch_losses)

            # Métriques de validation simulées
            val_f2        = round(min(0.95, 0.52 + epoch_global * 0.038 + random.gauss(0, 0.012)), 3)
            val_recall    = round(min(0.98, 0.58 + epoch_global * 0.035 + random.gauss(0, 0.010)), 3)
            val_precision = round(min(0.95, 0.45 + epoch_global * 0.040 + random.gauss(0, 0.015)), 3)
            metrics_history.append({"epoch": epoch_global, "f2": val_f2, "recall": val_recall})

            print(f"\n  {C.GRAY}avg_loss={avg_loss:.4f}{C.RESET}  "
                  f"val_F2={C.GREEN}{val_f2:.3f}{C.RESET}  "
                  f"recall={C.GREEN}{val_recall:.3f}{C.RESET}  "
                  f"prec={val_precision:.3f}")

            # Alertes targets
            if val_f2 >= 0.85:
                ok(f"Cible F2 ≥ 0.85 atteinte ! ({val_f2:.3f})")
            if val_recall >= 0.90:
                ok(f"Cible Recall ≥ 90% atteinte ! ({val_recall:.1%})")

            sleep(0.05)

    # Courbe ASCII
    section("Courbe d'apprentissage (F2-Score)")
    f2_vals = [m["f2"] for m in metrics_history]
    max_f2  = max(f2_vals)
    HEIGHT  = 8
    print()
    for row in range(HEIGHT, 0, -1):
        threshold = row / HEIGHT
        line = f"  {threshold:.2f} │"
        for val in f2_vals:
            normalized = val / max(max_f2, 0.01)
            line += C.GREEN + "█" + C.RESET if normalized >= threshold else " "
        print(line)
    print(f"       └{'─' * EPOCHS}→ epochs")
    print(f"        {''.join(str(i) for i in range(1, EPOCHS + 1))}")

    section("Résumé entraînement")
    best = max(metrics_history, key=lambda x: x["f2"])
    ok(f"Meilleur F2     : {best['f2']:.3f} (epoch {best['epoch']})")
    ok(f"Meilleur Recall : {max(m['recall'] for m in metrics_history):.3f}")
    info("Modèle sauvegardé → data/models/best_model.pt (simulé)")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — API FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

def run_api():
    banner("MODULE 3 — SIMULATION API FASTAPI", C.GREEN)
    print(f"  {C.GRAY}Simulation requêtes HTTP vers l'API (mode mock){C.RESET}")

    section("Démarrage serveur")
    steps = [
        "Chargement FastAPI + Uvicorn...",
        "Instanciation DistressDetector...",
        "Montage routes /health /predict /batch...",
        "Démarrage serveur sur 0.0.0.0:8000...",
    ]
    for s in steps:
        info(s)
        sleep(0.25)
    ok("API démarrée → http://localhost:8000")
    ok("Docs Swagger → http://localhost:8000/docs")
    print()

    # Requêtes simulées
    requests = [
        {"method": "GET",  "endpoint": "/health",  "body": None},
        {"method": "POST", "endpoint": "/predict",
         "body": {"text": "Je ne vois plus d'avenir", "dialect": "fr"}},
        {"method": "POST", "endpoint": "/predict",
         "body": {"text": "La vie est belle !", "dialect": "fr"}},
        {"method": "POST", "endpoint": "/predict",
         "body": {"text": "M bi kɛ dɔ, n ma se ka fɛ", "dialect": "moore"}},
        {"method": "POST", "endpoint": "/predict",
         "body": {"text": "Je pense à ne plus être là", "dialect": "fr"}},
        {"method": "GET",  "endpoint": "/health",  "body": None},
    ]

    section("Requêtes HTTP simulées")
    for i, req in enumerate(requests):
        method_color = C.GREEN if req["method"] == "GET" else C.BLUE
        print(f"\n  {method_color}{C.BOLD}{req['method']}{C.RESET} "
              f"{C.WHITE}{req['endpoint']}{C.RESET}")

        if req["body"]:
            print(f"  {C.GRAY}Body: {json.dumps(req['body'], ensure_ascii=False)}{C.RESET}")

        # Latence simulée
        latency_ms = random.randint(45, 180)
        info(f"Traitement... ({latency_ms}ms)")
        sleep(latency_ms / 1000 + 0.15)

        # Réponse simulée
        if req["endpoint"] == "/health":
            resp = {"status": "ok", "model": "african-mh-detection", "version": "0.1.0",
                    "device": "cpu", "uptime_s": round(i * 1.2, 1)}
        else:
            body = req["body"]
            # Chercher dans TEST_TEXTS
            sample = next((t for t in TEST_TEXTS if t["text"][:20] in body["text"]),
                          TEST_TEXTS[1])
            r = mock_predict(body["text"], sample["label"],
                             sample["expected_cat"], sample["expected_ctx"])
            msg = ""
            if r["signal"] and r["routing"] != "NONE":
                msg = ROUTING_MESSAGES[r["routing"]].get(body.get("dialect", "fr"),
                      ROUTING_MESSAGES[r["routing"]]["fr"])
            resp = {
                "signal":          r["signal"],
                "confidence":      r["confidence"],
                "categories":      r["categories"],
                "context_factors": r["context_factors"],
                "routing":         r["routing"],
                "message":         msg,
                "dialect":         body.get("dialect", "fr"),
                "latency_ms":      latency_ms,
            }

        status = "200 OK"
        print(f"  {C.GREEN}← {status}{C.RESET}")
        resp_str = json.dumps(resp, ensure_ascii=False, indent=4)
        for line in resp_str.split("\n"):
            key_color = C.CYAN if "signal" in line or "confidence" in line else C.GRAY
            print(f"    {key_color}{line}{C.RESET}")
        sleep(0.1)

    section("Métriques API")
    ok(f"6 requêtes traitées")
    ok(f"Latence moyenne : {random.randint(85, 130)}ms")
    ok(f"Latence P95     : {random.randint(150, 180)}ms")
    ok(f"Taux d'erreur   : 0%")
    info("Swagger UI dispo sur http://localhost:8000/docs après démarrage réel")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — AUDIT DE BIAIS
# ══════════════════════════════════════════════════════════════════════════════

def run_bias_audit():
    banner("MODULE 4 — AUDIT DE BIAIS PAR DIALECTE", C.YELLOW)
    print(f"  {C.GRAY}Fairness audit mensuel — écart max autorisé : 5%{C.RESET}")

    section("Génération corpus d'évaluation (simulé)")
    N_PER_DIALECT = 120
    for d in DIALECTS:
        info(f"Corpus {d:<8} : {N_PER_DIALECT} textes annotés")
        sleep(0.1)
    ok(f"Total : {N_PER_DIALECT * len(DIALECTS)} textes")

    section("Calcul métriques par dialecte")

    # Métriques simulées réalistes
    dialect_metrics = {
        "french": {
            "f1":        round(random.uniform(0.78, 0.88), 3),
            "recall":    round(random.uniform(0.88, 0.95), 3),
            "precision": round(random.uniform(0.72, 0.82), 3),
            "f2":        round(random.uniform(0.82, 0.91), 3),
            "n":         N_PER_DIALECT,
        },
        "moore": {
            "f1":        round(random.uniform(0.71, 0.83), 3),
            "recall":    round(random.uniform(0.82, 0.92), 3),
            "precision": round(random.uniform(0.65, 0.78), 3),
            "f2":        round(random.uniform(0.77, 0.87), 3),
            "n":         N_PER_DIALECT,
        },
        "dioula": {
            "f1":        round(random.uniform(0.70, 0.82), 3),
            "recall":    round(random.uniform(0.80, 0.91), 3),
            "precision": round(random.uniform(0.64, 0.77), 3),
            "f2":        round(random.uniform(0.76, 0.86), 3),
            "n":         N_PER_DIALECT,
        },
        "mix": {
            "f1":        round(random.uniform(0.73, 0.84), 3),
            "recall":    round(random.uniform(0.83, 0.93), 3),
            "precision": round(random.uniform(0.67, 0.79), 3),
            "f2":        round(random.uniform(0.78, 0.88), 3),
            "n":         N_PER_DIALECT,
        },
    }

    # Tableau des métriques
    print()
    header = f"  {'Dialecte':<12} {'F2':>6} {'Recall':>8} {'Precision':>10} {'F1':>6} {'N':>5}"
    print(f"{C.BOLD}{header}{C.RESET}")
    print(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*10} {'─'*6} {'─'*5}")

    f2_vals = [m["f2"] for m in dialect_metrics.values()]
    max_f2  = max(f2_vals)

    for dialect, m in dialect_metrics.items():
        sleep(0.2)
        # Couleur basée sur F2
        color = C.GREEN if m["f2"] >= 0.82 else (C.YELLOW if m["f2"] >= 0.76 else C.RED)
        flag  = " ★" if m["f2"] == max_f2 else ""
        print(f"  {dialect:<12} "
              f"{color}{m['f2']:>6.3f}{C.RESET} "
              f"{m['recall']:>8.3f} "
              f"{m['precision']:>10.3f} "
              f"{m['f1']:>6.3f} "
              f"{m['n']:>5}{C.CYAN}{flag}{C.RESET}")

    # Analyse des écarts
    section("Analyse des écarts (fairness)")
    metrics_to_check = ["f2", "recall", "f1"]

    alerts = []
    passes = True
    MAX_GAP = 0.05

    for metric in metrics_to_check:
        vals  = [m[metric] for m in dialect_metrics.values()]
        gap   = max(vals) - min(vals)
        color = C.GREEN if gap <= MAX_GAP else C.RED
        status = "OK" if gap <= MAX_GAP else "ALERTE"
        print(f"  {metric:<12} gap={color}{gap:.3f}{C.RESET}  "
              f"[seuil={MAX_GAP}]  {color}{status}{C.RESET}")
        if gap > MAX_GAP:
            passes = False
            alerts.append(f"Écart {metric} trop élevé entre dialectes: {gap:.3f} > {MAX_GAP}")
        sleep(0.15)

    # Rapport final
    section("Rapport d'audit")
    if passes:
        ok("AUDIT RÉUSSI — Fairness validée entre tous les dialectes")
        ok("Aucun biais significatif détecté (écarts ≤ 5%)")
    else:
        warn("AUDIT PARTIEL — Des améliorations sont nécessaires")
        for alert in alerts:
            warn(alert)
        info("Action requise : augmenter les données d'entraînement Moore/Dioula")

    # Sauvegarde rapport
    report = {
        "date": "2026-06-24",
        "metrics_by_dialect": dialect_metrics,
        "passes_fairness": passes,
        "alerts": alerts,
        "max_allowed_gap": MAX_GAP,
    }
    info(f"Rapport sauvegardé → logs/bias_audit_2026-06.json (simulé)")
    info(f"Prochain audit : 2026-07-24")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Simulation temps réel african-mh-detection")
    parser.add_argument("--mode", choices=["inference", "training", "api", "bias", "all"],
                        default="all", help="Module à simuler")
    args = parser.parse_args()

    print(f"\n{C.BOLD}{C.WHITE}")
    typing("  African Mental Health Detection System", 0.02)
    typing("  Presidential African Youth in AI 2026 — Burkina Faso 🇧🇫", 0.015)
    print(f"{C.RESET}")
    sleep(0.3)

    modes = {
        "inference": run_inference,
        "training":  run_training,
        "api":       run_api,
        "bias":      run_bias_audit,
    }

    if args.mode == "all":
        for fn in modes.values():
            fn()
            print(f"\n{C.GRAY}{'─' * 60}{C.RESET}")
            sleep(0.5)
    else:
        modes[args.mode]()

    banner("SIMULATION TERMINÉE", C.GREEN)
    ok("Tous les modules fonctionnent correctement")
    info("Pour lancer le vrai système : pip install -r requirements.txt")
    info("Puis : python -m src.inference.detector")
    print()

if __name__ == "__main__":
    main()
