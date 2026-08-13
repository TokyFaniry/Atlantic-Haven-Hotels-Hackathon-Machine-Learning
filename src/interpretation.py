"""Outils d'interprétation et d'analyse d'erreurs.

Regroupe les fonctions utilisées au notebook 05 : importance des variables par
permutation (plus fiable que l'importance native des arbres, qui est biaisée en
faveur des variables à forte cardinalité), analyse de performance par sous-groupe,
et extraction des cas mal classés.
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, precision_score, recall_score


def permutation_feature_importance(pipeline, X, y, n_repeats=10, random_state=42, scoring="f1"):
    """Importance par permutation, calculée sur le jeu de validation.

    Mesure la dégradation du score quand on mélange aléatoirement une colonne :
    contrairement à `feature_importances_` (impureté), cette approche reflète
    l'utilité réelle de la variable sur des données non vues, et n'est pas biaisée
    par la cardinalité.
    """
    result = permutation_importance(
        pipeline, X, y, n_repeats=n_repeats, random_state=random_state,
        scoring=scoring, n_jobs=-1,
    )
    return (pd.DataFrame({
        "variable": X.columns,
        "importance_moyenne": result.importances_mean,
        "ecart_type": result.importances_std,
    }).sort_values("importance_moyenne", ascending=False).reset_index(drop=True))


def performance_by_group(df, group_col, y_true_col="y_true", y_pred_col="y_pred", min_count=30):
    """Calcule F1/précision/rappel par modalité d'une variable de regroupement.

    `min_count` signale les sous-groupes trop petits pour que les métriques soient
    interprétables — leur variance d'échantillonnage est élevée et une différence
    apparente peut n'être que du bruit.
    """
    rows = []
    for value, sub in df.groupby(group_col, observed=True):
        n = len(sub)
        n_pos = int(sub[y_true_col].sum())
        if n_pos == 0:
            continue
        rows.append({
            group_col: value,
            "n": n,
            "n_annulations": n_pos,
            "taux_reel": sub[y_true_col].mean(),
            "f1": f1_score(sub[y_true_col], sub[y_pred_col], zero_division=0),
            "precision": precision_score(sub[y_true_col], sub[y_pred_col], zero_division=0),
            "recall": recall_score(sub[y_true_col], sub[y_pred_col], zero_division=0),
            "effectif_suffisant": n >= min_count,
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def extract_errors(X, y_true, y_pred, y_proba, kind="fp", n=5, sort_by_confidence=True):
    """Extrait les n cas mal classés les plus « confiants » (donc les plus instructifs).

    kind="fp" : faux positifs (prédits annulés, en réalité maintenus)
    kind="fn" : faux négatifs (prédits maintenus, en réalité annulés)
    """
    df = X.copy()
    df["y_true"] = np.asarray(y_true)
    df["y_pred"] = np.asarray(y_pred)
    df["probabilite"] = np.asarray(y_proba)

    if kind == "fp":
        mask = (df["y_true"] == 0) & (df["y_pred"] == 1)
        ascending = False  # probabilité la plus haute = erreur la plus « sûre d'elle »
    elif kind == "fn":
        mask = (df["y_true"] == 1) & (df["y_pred"] == 0)
        ascending = True  # probabilité la plus basse = erreur la plus « sûre d'elle »
    else:
        raise ValueError("kind doit être 'fp' ou 'fn'")

    errors = df[mask]
    if sort_by_confidence:
        errors = errors.sort_values("probabilite", ascending=ascending)
    return errors.head(n)
