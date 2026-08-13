"""Configuration centrale du projet Atlantic Haven Hotels.

Toutes les constantes partagées entre les notebooks (chemins, graine aléatoire,
listes de colonnes) sont définies ici pour éviter la duplication et garantir
la cohérence entre les étapes.
"""
from pathlib import Path

# --- Reproductibilité -------------------------------------------------------
RANDOM_STATE = 42

# --- Chemins -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "ressources"
TRAIN_PATH = DATA_DIR / "reservations_train.csv"
TEST_PATH = DATA_DIR / "reservations_test.csv"
DATA_DICT_PATH = DATA_DIR / "data_dictionary.csv"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TARGET_COL = "reservation_annulee"
ID_COL = "reservation_id"
DATE_COLS = ["date_reservation", "date_arrivee"]

# --- Colonnes écartées pour la baseline (redondantes ou non exploitables) ---
# Justification détaillée dans notebooks/01_eda.ipynb (§1.4) et
# notebooks/02_baseline.ipynb (§2.2).
DROP_COLS = ["reservation_id", "date_reservation", "date_arrivee",
             "region_hotel", "ville", "hotel_id"]

NUM_COLS = ["categorie_hotel", "delai_reservation_jours", "nuits", "adultes", "enfants",
            "chambres", "prix_moyen_nuit_eur", "remise_pct", "montant_total_eur",
            "reservations_passees", "annulations_passees", "demandes_speciales",
            "modifications_reservation", "jours_liste_attente", "evenement_majeur",
            "haute_saison_regionale", "arrivee_weekend"]

CAT_COLS = ["type_destination", "segment_client", "marche_origine", "canal_reservation",
            "moyen_transport", "formule_repas", "tarif_remboursable", "type_acompte",
            "client_type", "agent_id"]

# --- Colonnes étendues après feature engineering (étape 4) -----------------
# Les nouvelles variables candidates sont ajoutées par src.features.add_engineered_features
# et src.features.GroupRelativePriceTransformer. L'ablation menée dans
# notebooks/04_feature_engineering.ipynb (§4.2) a montré qu'un sous-ensemble ciblé
# de 3 variables apporte un gain réel et reproductible, alors que le jeu complet
# (fortement redondant avec les variables brutes) n'en apporte pas — voire nuit
# légèrement par sur-paramétrage. On ne retient donc que ce sous-ensemble validé :
FE_NUM_COLS = ["personnes_par_chambre", "prix_relatif_destination", "taux_annulation_passee"]

NUM_COLS_FE = NUM_COLS + FE_NUM_COLS
CAT_COLS_FE = list(CAT_COLS)  # aucune variable catégorielle supplémentaire retenue

# Ensemble complet des variables candidates testées en ablation (non retenu en
# production, conservé ici pour la traçabilité de la démarche expérimentale).
FE_NUM_COLS_CANDIDATES = [
    "mois_arrivee", "jour_semaine_arrivee", "delai_log",
    "personnes_totales", "personnes_par_chambre", "nuits_par_chambre", "a_enfants",
    "prix_total_par_personne", "deja_annule_avant", "taux_annulation_passee",
    "client_nouveau_sans_historique", "prix_relatif_destination",
]
FE_CAT_COLS_CANDIDATES = ["delai_bin"]
