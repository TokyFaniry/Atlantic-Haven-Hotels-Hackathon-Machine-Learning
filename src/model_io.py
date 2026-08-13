"""Sauvegarde et chargement des pipelines entraînés (joblib).

Permet aux notebooks en aval (04 feature engineering, 05 interprétation,
06 submission) de réutiliser un modèle entraîné dans un notebook précédent
sans tout ré-entraîner, tout en gardant chaque notebook exécutable seul
(si l'artefact n'existe pas encore, le notebook amont peut être relancé).
"""
import json
import joblib

from .config import ARTIFACTS_DIR


def save_model(pipeline, name, metadata=None):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{name}.joblib"
    joblib.dump(pipeline, path)
    if metadata is not None:
        meta_path = ARTIFACTS_DIR / f"{name}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    print(f"Modèle sauvegardé : {path}")
    return path


def load_model(name):
    path = ARTIFACTS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Artefact introuvable : {path}. Exécutez d'abord le notebook qui l'entraîne."
        )
    return joblib.load(path)
