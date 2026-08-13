"""Feature engineering — création de variables dérivées.

Deux catégories de fonctions ici :

1. `add_engineered_features` : transformations **déterministes ligne par ligne**
   (dates, composition du séjour, prix, historique). Aucune fuite possible car
   chaque valeur ne dépend que des colonnes de la même ligne.

2. `GroupRelativePriceTransformer` : une transformation qui dépend de statistiques
   **apprises sur le train** (moyenne de prix par type de destination). Implémentée
   comme un transformer scikit-learn (`fit`/`transform`) pour être insérée dans un
   `Pipeline` et garantir que ces statistiques ne sont jamais calculées sur la
   validation ou le test.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def add_engineered_features(df):
    """Ajoute les variables dérivées à un DataFrame brut (avant `prepare_features`).

    Doit être appelé sur les données **avant** la suppression des colonnes de dates
    brutes, puisque `mois_arrivee` et `jour_semaine_arrivee` en dépendent.
    """
    df = df.copy()

    # --- Temporel ---------------------------------------------------------
    df["mois_arrivee"] = df["date_arrivee"].dt.month
    df["jour_semaine_arrivee"] = df["date_arrivee"].dt.dayofweek
    df["delai_log"] = np.log1p(df["delai_reservation_jours"])
    bins = [-1, 7, 30, 90, 180, np.inf]
    labels = ["0-7j", "8-30j", "31-90j", "91-180j", "181j+"]
    df["delai_bin"] = pd.cut(df["delai_reservation_jours"], bins=bins, labels=labels).astype(str)

    # --- Composition du séjour ---------------------------------------------
    enfants_filled = df["enfants"].fillna(0)
    df["personnes_totales"] = df["adultes"] + enfants_filled
    df["personnes_par_chambre"] = df["personnes_totales"] / df["chambres"].replace(0, np.nan)
    df["nuits_par_chambre"] = df["nuits"] / df["chambres"].replace(0, np.nan)
    df["a_enfants"] = (enfants_filled > 0).astype(int)

    # --- Prix ----------------------------------------------------------------
    df["prix_total_par_personne"] = df["montant_total_eur"] / df["personnes_totales"].replace(0, np.nan)

    # --- Historique client ---------------------------------------------------
    df["deja_annule_avant"] = (df["annulations_passees"] > 0).astype(int)
    df["taux_annulation_passee"] = (
        df["annulations_passees"] / df["reservations_passees"].replace(0, np.nan)
    ).fillna(0)
    df["client_nouveau_sans_historique"] = (df["reservations_passees"] == 0).astype(int)

    return df


class GroupRelativePriceTransformer(BaseEstimator, TransformerMixin):
    """Ajoute `prix_relatif_destination` = prix_moyen_nuit_eur / moyenne du groupe.

    La moyenne par groupe (`type_destination` par défaut) est apprise en `fit()`
    exclusivement sur les données passées (le train), puis simplement réutilisée
    en `transform()` sur la validation et le test — aucune fuite d'information.
    """

    def __init__(self, group_col="type_destination", value_col="prix_moyen_nuit_eur"):
        self.group_col = group_col
        self.value_col = value_col

    def fit(self, X, y=None):
        self.global_mean_ = X[self.value_col].mean()
        self.group_means_ = X.groupby(self.group_col)[self.value_col].mean()
        return self

    def transform(self, X):
        X = X.copy()
        means = X[self.group_col].map(self.group_means_).fillna(self.global_mean_)
        X["prix_relatif_destination"] = X[self.value_col] / means
        return X
