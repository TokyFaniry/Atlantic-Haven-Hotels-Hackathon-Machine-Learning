"""Prétraitement des features.

Toutes les statistiques (médianes, modalités) sont apprises exclusivement via
`.fit()` sur le train, au sein d'un `ColumnTransformer` scikit-learn, pour
garantir qu'aucune information de la validation ou du test ne fuite dans
l'apprentissage (exigence explicite du sujet).
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DROP_COLS, NUM_COLS, CAT_COLS


def prepare_features(df):
    """Supprime les colonnes non exploitables et normalise `agent_id` manquant.

    `agent_id` est structurellement vide pour les canaux de réservation directs
    (site hôtel, téléphone, entreprise) — ce n'est pas une valeur manquante au
    sens statistique, on la remplace donc par une catégorie explicite "direct"
    plutôt que de l'imputer aveuglément (cf. notebooks/01_eda.ipynb §1.3).
    """
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    if "agent_id" in df.columns:
        df["agent_id"] = df["agent_id"].fillna("direct")
    return df


def build_preprocessor(num_cols=None, cat_cols=None):
    """Construit le ColumnTransformer commun à tous les modèles.

    - Numérique : imputation par la médiane + standardisation.
    - Catégoriel : imputation par une catégorie "Inconnu" + one-hot encoding
      (`handle_unknown="ignore"` pour gérer les modalités jamais vues au train,
      qui pourraient apparaître dans le test — cf. Q6 du README).
    """
    num_cols = num_cols if num_cols is not None else NUM_COLS
    cat_cols = cat_cols if cat_cols is not None else CAT_COLS

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Inconnu")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ])


def make_pipeline(estimator, num_cols=None, cat_cols=None):
    """Assemble le ColumnTransformer commun et un estimateur donné dans un seul Pipeline.

    Garantit que tous les modèles comparés (étape 3) reçoivent exactement le même
    prétraitement — seule la partie "estimateur" varie — pour une comparaison équitable.
    """
    return Pipeline([
        ("preprocess", build_preprocessor(num_cols, cat_cols)),
        ("model", estimator),
    ])
