"""Chargement des données et split de validation temporel.

Le split temporel est central au protocole de validation exigé par le sujet
(§2 EDA / §3 modélisation) : le jeu de test est postérieur au train, donc
toute validation interne doit respecter cet ordre chronologique.
"""
import pandas as pd

from .config import TRAIN_PATH, TEST_PATH, DATA_DICT_PATH, TARGET_COL, DATE_COLS


def load_train_test():
    """Charge les jeux d'entraînement, de test et le dictionnaire de données."""
    train = pd.read_csv(TRAIN_PATH, parse_dates=DATE_COLS)
    test = pd.read_csv(TEST_PATH, parse_dates=DATE_COLS)
    data_dict = pd.read_csv(DATA_DICT_PATH)
    return train, test, data_dict


def temporal_split(train, target_col=TARGET_COL, val_frac=0.2, date_col="date_reservation"):
    """Sépare le train en (train interne, validation) en respectant l'ordre chronologique.

    Les `val_frac` dernières observations (triées par `date_col`) sont réservées à la
    validation, ce qui simule le contexte réel : prédire des réservations futures à
    partir de réservations passées, comme c'est le cas entre `reservations_train.csv`
    et `reservations_test.csv`.

    Retourne (X_tr, X_val, y_tr, y_val, split_date).
    """
    # Colonnes d'exploration ajoutées ponctuellement en EDA (ex: delai_bin) à ignorer
    # si elles sont présentes, pour ne pas polluer les features en aval.
    exploratory_cols = ["delai_bin", "mois_reservation", "mois_arrivee", "resa_bin"]
    train_clean = train.drop(columns=[c for c in exploratory_cols if c in train.columns])

    train_sorted = train_clean.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(train_sorted) * (1 - val_frac))
    split_date = train_sorted.loc[split_idx, date_col]

    X = train_sorted.drop(columns=[target_col])
    y = train_sorted[target_col]

    X_tr, X_val = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_tr, y_val = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

    return X_tr, X_val, y_tr, y_val, split_date
