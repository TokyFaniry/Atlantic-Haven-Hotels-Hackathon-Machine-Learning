# **Rapport de Projet — Atlantic Haven Hotels**

## **Examen Final Machine Learning & Data Science — M1**

Réalisé au sein de **ISPM — Madagascar** ([www.ispm-edu.com](https://www.ispm-edu.com))

---

### **1. Informations sur le Groupe**

#### Membre 1

- nom : RAVELONARIVO
- prénom(s) : Fanantenana Mickaël
- classe : M1 — Machine Learning & Data Science
- numéro : 28
- rôle : Analyste données — EDA et préparation (notebook 01)

#### Membre 2

- nom : IALISOA
- prénom(s) : Iris Fifaliana
- classe : M1 — Machine Learning & Data Science
- numéro : 33
- rôle : Développeuse ML — baseline et protocole de validation temporelle (notebook 02)

#### Membre 3

- nom : RAZAFIMBELO
- prénom(s) : Toky Faniry
- classe : M1 — Machine Learning & Data Science
- numéro : 34
- rôle : Responsable de la modélisation — comparaison des modèles et choix du seuil (notebook 03), intégration technique et dépôt GitHub

#### Membre 4

- nom : RAMEFIARISON
- prénom(s) : Fabio Fandresena
- classe : M1 — Machine Learning & Data Science
- numéro : 35
- rôle : Développeur — feature engineering et validation du gain (notebook 04)

#### Membre 5

- nom : RAKOTOARIMANANA
- prénom(s) : Tojo Ny Aina
- classe : M1 — Machine Learning & Data Science
- numéro : 38
- rôle : Analyste — interprétation et analyse d'erreurs (notebook 05)

#### Membre 6

- nom : FANAMBIHARINDRAINY
- prénom(s) : Schenyolla Anderssen
- classe : M1 — Machine Learning & Data Science
- numéro : 37
- rôle : Responsable soumission finale (notebook 06) et présentatrice de la vidéo

---

### **2. Résumé du Travail**

#### Problématique

Atlantic Haven Hotels exploite des établissements dans dix régions italiennes couvrant des destinations très
variées (urbaines, balnéaires, montagneuses, insulaires, rurales). Une annulation tardive laisse une chambre
inoccupée et perturbe la planification opérationnelle (staffing, restauration, gestion de l'inventaire). L'enjeu
n'est pas seulement de détecter les annulations, mais de le faire **suffisamment tôt et sans pénaliser** les
nombreux clients dont le profil ressemble statistiquement à un client à risque mais qui honoreront leur séjour.

#### Méthodologie adoptée

1. **EDA** : cible déséquilibrée (25,8 % d'annulations), analyse des valeurs manquantes (dont `agent_id`,
   structurellement vide pour les canaux directs et non un vrai defect), des variables catégorielles à forte
   cardinalité, des outliers (conservés, jugés plausibles) et des relations temporelles.
2. **Protocole de validation temporel obligatoire** : split chronologique (dernier 20 % du train, trié par
   `date_reservation`, sert de validation) au lieu d'une validation croisée aléatoire, car le jeu de test est
   postérieur au train.
3. **Baseline** : régression logistique (`class_weight="balanced"`), tous les prétraitements (imputation,
   encodage) appris uniquement sur le train interne.
4. **Modélisation** : comparaison de trois familles — régression logistique, Random Forest, XGBoost — réglées
   par validation croisée temporelle (`TimeSeriesSplit`), avec recherche du seuil de décision maximisant le F1
   pour chaque modèle.
5. **Feature engineering** : construction de variables dérivées (temporelles, séjour/prix, historique client),
   testées d'abord en bloc puis par ablation par thème pour isoler un gain réel et non un artefact de bruit.
6. **Interprétation** : importance par permutation, analyse qualitative des faux positifs / faux négatifs,
   performance par sous-groupe, et traduction des probabilités en recommandation opérationnelle graduée.

#### Résultats obtenus

Le modèle final retenu est **XGBoost** (sans les variables de feature engineering, qui n'apportent pas de gain
mesurable sur ce modèle — voir §5 Q3), avec un **F1-score de 0,4814** sur la validation temporelle (seuil
optimal = 0,425), pour un ROC-AUC de 0,652. Découverte importante : le jeu complet de variables candidates
**dégrade** légèrement les performances (redondance/bruit), alors qu'un sous-ensemble minimal ciblé apporte un
gain reproductible mais uniquement sur le modèle linéaire — les modèles arborescents reconstruisent déjà ces
interactions par eux-mêmes.

#### Mots-clés

Classification binaire déséquilibrée, annulation hôtelière, validation temporelle, F1-score, feature
engineering, XGBoost, analyse d'erreurs, priorisation opérationnelle.

---

### **3. Contenu du Repository**

- **`notebook.ipynb`** : notebook unique, fusion linéaire des 6 étapes (EDA → soumission), exécutable de bout
  en bout depuis un noyau vierge ;
- **`notebooks/01_eda.ipynb` → `06_submission.ipynb`** : les mêmes 6 étapes en notebooks séparés, pour une
  lecture modulaire ;
- **`src/`** : modules partagés (`config.py`, `data.py`, `preprocessing.py`, `features.py`, `evaluation.py`,
  `interpretation.py`, `model_io.py`) — chemins, split temporel, pipelines scikit-learn, métriques, sauvegarde
  des modèles ;
- **`ressources/`** : `reservations_train.csv`, `reservations_test.csv`, `data_dictionary.csv` (fournis par
  l'énoncé) ;
- **`artifacts/`** : modèles entraînés sauvegardés (`.joblib` + métadonnées `.json`) ;
- **`submission.csv`** : prédictions sur `reservations_test.csv` ;
- **`requirements.txt`** : dépendances nécessaires à la reproduction du projet.

**🔗 Liens utiles :**

- [**LIEN VERS LA VIDÉO DE PRÉSENTATION**](https://drive.google.com/file/d/1TVusxh6tuOfyEPDVy-hyuPXEzakxjbFX/view?usp=drivesdk)
- [Dépôt GitHub](https://github.com/TokyFaniry/Atlantic-Haven-Hotels-Hackathon-Machine-Learning)

---

### **4. Résultats de Modélisation**

Résultats comparés sur le **même jeu de validation temporel** (dernières 1 600 réservations du train, à partir
du 2024-11-28), au seuil optimisant le F1 pour chaque modèle :

| Modèle | Paramètres principaux | F1-score | Précision | Rappel | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Régression logistique — baseline | `class_weight="balanced"`, seuil = 0,30 | 0,4727 | 0,3183 | 0,9184 | 0,6532 |
| Random Forest | réglé par `RandomizedSearchCV` + `TimeSeriesSplit`, seuil = 0,45 | 0,4761 | 0,3433 | 0,7762 | 0,6530 |
| XGBoost | `n_estimators=150, max_depth=4, learning_rate=0.01, subsample=0.85, colsample_bytree=0.85`, seuil = 0,425 | **0,4814** | 0,3394 | 0,8275 | 0,6523 |
| **Modèle final (retenu)** | XGBoost ci-dessus, réentraîné sur les 8 000 lignes du train complet | **0,4814** (estimé) | 0,3394 | 0,8275 | 0,6523 |

**Seuil de décision retenu :** 0,425 (maximise le F1 sur la validation temporelle).

**Justification du choix du modèle final :** XGBoost domine légèrement les deux autres modèles sur le F1 au
seuil optimal, et capture nativement les relations non linéaires observées en EDA (ex. le taux d'annulation
qui double au-delà d'un certain délai de réservation) ainsi que les interactions entre variables, sans
nécessiter de feature engineering explicite. Le feature engineering testé (§4) n'apporte pas de gain
reproductible sur ce modèle précis (contrairement à la régression logistique) — XGBoost reconstruit déjà ces
interactions par ses splits successifs. Le modèle est conservé sans ces variables pour la simplicité du
pipeline. Sa robustesse a été vérifiée sur 5 graines aléatoires différentes (F1 = 0,482 ± 0,001).

---

### **5. Réponses aux Questions d'Analyse**

#### **Q1. Pourquoi utilise-t-on principalement le F1-score plutôt que l'accuracy pour cette tâche ?**

La cible est déséquilibrée (25,8 % d'annulations). Un modèle qui prédit systématiquement « pas d'annulation »
obtiendrait déjà ~74 % d'accuracy sans détecter une seule annulation — ce qui est inutile pour l'hôtel. Le
F1-score, moyenne harmonique de la précision et du rappel **sur la classe minoritaire « annulation »**, oblige
le modèle à identifier réellement les annulations plutôt qu'à se reposer sur le déséquilibre des classes.

#### **Q2. Dans ce contexte, qu'est-ce qui est le plus grave : un faux positif ou un faux négatif ?**

- **Faux positif** : une réservation qui sera honorée, mais que le modèle signale comme « à risque ». Coût
  faible **si** l'action déclenchée est non punitive (message de confirmation, ajustement de planning
  interne) ; coût élevé si elle se traduit par une action agressive envers le client (annulation préventive,
  demande d'acompte supplémentaire).
- **Faux négatif** : une annulation réelle non anticipée. Coût opérationnel direct — chambre laissée vacante
  sans plan de repli, perte de revenu, planification faussée.

Dans le contexte de cet examen (métrique F1, mission de ne pas pénaliser les clients fidèles), le **faux
négatif est le plus coûteux opérationnellement**, mais seulement tant que le traitement des faux positifs
reste non punitif — d'où le choix d'un seuil qui privilégie le rappel (0,83) et une utilisation graduée des
probabilités plutôt qu'une décision binaire brute (voir Q7).

#### **Q3. Quelles variables créées par feature engineering ont le plus amélioré votre modèle par rapport à la régression logistique de référence ?**

Le sous-ensemble **« minimal ciblé »** — `personnes_par_chambre` (densité d'occupation), `prix_relatif_destination`
(prix comparé à la moyenne de la destination) et `taux_annulation_passee` (historique client relatif, pas en
valeur brute) — améliore la régression logistique de **F1 = 0,4537 → 0,4578 (+0,0041)** et **ROC-AUC = 0,6532 →
0,6563 (+0,0031)** à seuil fixe (0,5). Le gain est réel et reproductible car ces variables encodent des
**ratios/relatifs** qu'un modèle linéaire ne peut pas reconstituer lui-même à partir des colonnes brutes. En
revanche, le même jeu **n'apporte aucun gain reproductible sur XGBoost** (Δ moyen < écart-type sur 5 graines),
car un modèle arborescent peut déjà approximer ces ratios via des splits successifs.

#### **Q4. Pourquoi un découpage aléatoire simple peut-il produire une évaluation trompeuse sur ce dataset ?**

Les données sont ordonnées dans le temps et le jeu de test représente des réservations **postérieures** à
l'entraînement. Une validation croisée aléatoire mélangerait passé et futur, autorisant le modèle à
« apprendre » indirectement des tendances futures et produisant un score optimiste, non représentatif de la
performance réelle en production (où l'on prédit toujours des réservations à venir à partir de données
passées). Le protocole adopté ici découpe chronologiquement : les 6 400 premières réservations (2023-01-01 →
2024-11-28) servent à l'entraînement interne, les 1 600 suivantes (2024-11-28 → 2025-05-24) à la validation —
un miroir direct de la relation train/test réelle du sujet.

#### **Q5. Quels profils ou scénarios de réservation sont les plus fréquemment associés aux annulations dans vos analyses ?**

- Réservations **sans acompte** (`type_acompte = "aucun"`) et à **tarif remboursable** — 91,3 % des vrais
  positifs ont un tarif remboursable, contre 21,5 % des vrais négatifs ;
- Délai de réservation long (réservations prises très à l'avance) ;
- Réservations effectuées via des canaux en ligne (plateforme) plutôt qu'en direct ;
- Clients avec un historique d'annulations passées plus élevé.

*(Ces facteurs décrivent des circonstances de vente et de réservation, et non des caractéristiques
intrinsèques d'une région ou d'une population — conformément à la mise en garde du sujet.)*

#### **Q6. Comment votre pipeline traite-t-il les valeurs manquantes et les catégories jamais observées pendant l'entraînement ?**

Toutes les statistiques d'imputation (médianes, catégorie `"Inconnu"`) sont apprises **uniquement sur le train
interne**, à l'intérieur d'un `Pipeline` scikit-learn, puis appliquées telles quelles à la validation et au
test — aucune fuite. Concrètement : `enfants`, `demandes_speciales` et `prix_moyen_nuit_eur` sont imputés par
la médiane ; `marche_origine` par une catégorie explicite `"Inconnu"` (pour ne pas fabriquer un signal
artificiel avec le mode) ; `agent_id` manquant est recodé en catégorie `"direct"`. Les catégories inédites (ex.
`canal_reservation = "assistant_vocal"`, apparue uniquement dans le test) sont gérées par
`OneHotEncoder(handle_unknown="ignore")`, qui les encode comme n'appartenant à aucune catégorie connue sans
provoquer d'erreur.

#### **Q7. Selon vous, quelle action l'hôtel devrait-il entreprendre lorsqu'une réservation en cours présente une forte probabilité d'annulation ?**

Une action **graduée et non punitive**, proportionnée à la tranche de risque :

| Tranche de probabilité | Action recommandée |
|---|---|
| Faible (< 0,30) | Parcours client standard |
| Modéré (0,30 – seuil) | E-mail de confirmation à J-7 (informations pratiques) |
| Élevé (seuil – 0,60) | Contact personnalisé, offre de service additionnel qui renforce l'engagement |
| Très élevé (> 0,60) | Ajustement de la planification interne (surbooking mesuré, liste d'attente) — invisible pour le client |

Avec une précision d'environ 35 %, deux réservations « à risque » sur trois seront en réalité honorées : toute
action pénalisante toucherait majoritairement des clients fidèles. L'intervention doit donc coûter peu quand
elle est appliquée à tort (message, ajustement interne) plutôt que de sanctionner le client.

#### **Q8. Votre modèle présente-t-il des performances comparables selon les régions ou les types de destination ?**

Non — le F1 varie de 0,395 (`urbaine_cotiere`) à 0,586 (`insulaire_mixte`) selon le type de destination, et de
0,306 (`entreprise`) à 0,549 (`agence`) selon le canal de réservation. Cet écart reflète cependant
essentiellement le **taux d'annulation de base** de chaque sous-groupe (un groupe où les annulations sont
rares est mécaniquement plus difficile à prédire avec un seuil unique) et la taille des effectifs, plutôt
qu'une défaillance ciblée du modèle sur une population donnée. Limite à noter : plusieurs sous-groupes ont un
effectif inférieur à 150 réservations en validation, ce qui rend ces estimations de F1 individuelles peu
précises.

#### **Q9. Analyse des erreurs**

**Faux positifs (5 exemples) :** des réservations qui cumulent les marqueurs statistiques de risque — délai
long (~41 jours en moyenne vs 35 pour les vrais négatifs), absence d'acompte (69,6 % vs 7,1 %), tarif
remboursable (88,6 % vs 21,5 %), canal en ligne — mais que le client a finalement honorées. Le modèle
n'est pas « en tort » sur le profil : l'intention réelle du client n'est simplement présente dans aucune
colonne du dataset.

**Faux négatifs (5 exemples) :** des annulations « surprises » sur des profils très fermes — acompte total
versé (52,7 % des cas), tarif non remboursable (70,3 %), délai plus court que la moyenne des vrais positifs
(39,6 j vs 49,1 j). Ce sont des cas où toutes les variables disponibles indiquaient un séjour ferme.

**Raisons possibles :** avec un ROC-AUC de ~0,65, le signal disponible dans les données de réservation est
modéré. Les erreurs proviennent largement de facteurs **exogènes non observables** : imprévus personnels ou
professionnels, problèmes de transport, changement de plans — aucune variable du dataset ne capture
l'intention réelle du client au-delà de proxies commerciaux (acompte, remboursabilité).

**Piste d'amélioration :** enrichir le dataset avec des signaux comportementaux post-réservation (consultation
de la page d'annulation, échanges avec le service client, non-réponse aux e-mails) ou des données
contextuelles externes (grèves de transport, météo, alertes sanitaires sur la période d'arrivée).

---

### **6. Conclusion et Recommandations**

Le modèle XGBoost final atteint un F1 de 0,4814 sur la validation temporelle, nettement au-dessus d'une
prédiction naïve, mais avec une précision limitée (~34 %) due au signal modéré disponible dans des données de
réservation seules (ROC-AUC ≈ 0,65). Le modèle prédit un taux d'annulation global (~65-67 %) très supérieur au
taux réel (~26-27 %) : c'est la conséquence assumée de l'optimisation du F1 (qui privilégie fortement le
rappel), et non un défaut de calibration à corriger — un seuil plus « réaliste » (~0,575) ferait chuter le F1
de 16 %. Le modèle doit donc être utilisé comme **outil de priorisation et de classement du risque**, pas
comme prédicteur individuel fiable.

**Recommandation opérationnelle finale :** intégrer les probabilités prédites dans un tableau de bord de
priorisation à 4 niveaux (voir Q7), avec des actions strictement non punitives pour les tranches faible/
modérée/élevée, et réserver les ajustements de planification interne (surbooking mesuré, liste d'attente) à
la tranche « très élevé ». Un réentraînement périodique est recommandé pour absorber l'évolution des canaux de
réservation (ex. apparition d'`assistant_vocal` dans le test) et limiter la dérive du modèle dans le temps.

---

### **7. Reproductibilité**

- **Version de Python :** 3.12.4
- **Principales bibliothèques et versions :** pandas 3.0.0, numpy 2.0.0, scikit-learn 1.8.0, xgboost 3.2.0,
  matplotlib, seaborn, joblib (voir `requirements.txt`)
- **Graine(s) aléatoire(s) :** `RANDOM_STATE = 42` (fixée de façon centralisée dans `src/config.py`,
  réutilisée dans tous les splits, modèles et recherches d'hyperparamètres)
- **Commande / procédure d'exécution :**
  ```bash
  pip install -r requirements.txt
  jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
  ```
  (ou exécution manuelle, noyau vierge, `Run All`)
- **Durée approximative d'entraînement :** quelques minutes sur poste standard (Random Forest et XGBoost
  réglés par recherche aléatoire sur validation croisée temporelle à 4 plis)
- **Environnement utilisé :** local (Windows)

---

### **8. Bibliographie**

- Documentation officielle scikit-learn (`Pipeline`, `ColumnTransformer`, `TimeSeriesSplit`,
  `permutation_importance`) — https://scikit-learn.org/stable/
- Documentation officielle XGBoost (`XGBClassifier`, `scale_pos_weight`) — https://xgboost.readthedocs.io/
- Sujet et dictionnaire de données fournis par l'ISPM (`ressources/data_dictionary.csv`)
- Outil d'IA générative utilisé : **Claude Code (Anthropic)**, pour la revue du respect du cahier des charges,
  la correction d'un bug de chemin de données (`src/config.py`), la relance du pipeline d'entraînement et la
  rédaction assistée de ce rapport à partir des résultats produits par le notebook.
