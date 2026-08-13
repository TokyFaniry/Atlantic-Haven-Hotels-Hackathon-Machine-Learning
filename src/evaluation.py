"""Fonctions d'évaluation communes à tous les modèles.

Centralise le calcul des métriques (F1, précision, rappel, ROC-AUC), la
matrice de confusion et la recherche de seuil optimal, pour garantir que
tous les modèles sont comparés exactement de la même façon (cf. §4 du
README, "Résultats de Modélisation").
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score, confusion_matrix)


def evaluate(y_true, y_proba, threshold=0.5, label="Modèle"):
    """Calcule les métriques standard pour un jeu de probabilités donné."""
    y_pred = (y_proba >= threshold).astype(int)
    results = {
        "modele": label,
        "seuil": threshold,
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    return results, y_pred


def plot_confusion(y_true, y_pred, title="Matrice de confusion"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Prédit: Maintenue", "Prédit: Annulée"],
                yticklabels=["Réel: Maintenue", "Réel: Annulée"])
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return cm


def threshold_curve(y_true, y_proba, thresholds=None):
    """Calcule F1/précision/rappel pour une grille de seuils de décision."""
    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 37)
    f1s, precisions, recalls = [], [], []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        f1s.append(f1_score(y_true, pred))
        precisions.append(precision_score(y_true, pred, zero_division=0))
        recalls.append(recall_score(y_true, pred))
    return thresholds, np.array(f1s), np.array(precisions), np.array(recalls)


def plot_threshold_curve(thresholds, f1s, precisions, recalls, title="Seuil de décision"):
    best_t = thresholds[np.argmax(f1s)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, f1s, label="F1-score", linewidth=2, color="#C44E52")
    ax.plot(thresholds, precisions, label="Précision", linestyle="--", color="#4C72B0")
    ax.plot(thresholds, recalls, label="Rappel", linestyle="--", color="#55A868")
    ax.axvline(best_t, color="gray", linestyle=":", label=f"Meilleur seuil F1 = {best_t:.2f}")
    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return best_t
