# Script vidéo — résumé à dire (3 à 5 minutes)

> Fabio absent : sa partie (feature engineering) est reprise par Tojo, juste avant l'interprétation.
> Gardez `dashboard/index.html` ouvert à l'écran et montrez la section correspondante au fil du discours.
> Parlez avec vos mots à partir de ces points — pas besoin d'apprendre par cœur.

---

## 1. Toky — Intro (0:00–0:25)
- Équipe Atlantic Haven Hotels, examen ML & Data Science.
- Mission : prédire l'annulation d'une réservation, sans pénaliser les clients qui vont honorer leur séjour.
- Rôles : Fanantenana (EDA), Iris (baseline), Toky (modélisation), Tojo (feature engineering +
  interprétation), Schenyolla (soumission + recommandation).

## 2. Fanantenana — EDA (0:25–1:05)
- Cible déséquilibrée : **25,8% d'annulations** → d'où le choix du F1-score plutôt que l'accuracy.
- `agent_id` manquant à 42% = normal, pas un défaut (vide pour les canaux directs).
- Le taux d'annulation **augmente avec le délai de réservation**.
- Validation **temporelle obligatoire** : test = données plus récentes que le train → split chronologique
  (6400 lignes train interne / 1600 validation), pas de validation croisée aléatoire.

## 3. Iris — Baseline (1:05–1:30)
- Régression logistique, prétraitements appris **uniquement sur le train**.
- **F1 = 0,4727** au seuil optimal (0,30) → score de référence à battre.

## 4. Toky — Modélisation (1:30–2:15)
- 3 modèles comparés au même seuil optimal : LogReg 0,4727 / Random Forest 0,4761 / **XGBoost 0,4814**
  (retenu).
- Réglage par validation croisée **temporelle** (`TimeSeriesSplit`).
- Seuil retenu **0,425** → rappel 0,83, précision 0,34 : on privilégie la détection des annulations,
  cohérent avec l'objectif F1 du sujet.

## 5. Tojo — Feature engineering (2:15–2:45)
- Variables créées : personnes/chambre, prix relatif à la destination, taux d'annulation passé.
- Gain réel sur la régression logistique (+0,0041 F1), **mais aucun gain sur XGBoost** (vérifié sur
  5 graines) — les arbres reconstruisent déjà ces rapports tout seuls.
- Décision : garder XGBoost **sans** ces variables pour le modèle final.

## 6. Tojo — Interprétation (2:45–3:30)
- Variables les plus importantes : `type_acompte`, `tarif_remboursable`, `canal_reservation` — des leviers
  commerciaux actionnables par l'hôtel.
- Faux positifs : clients "à risque" sur le papier (pas d'acompte, remboursable) qui honorent quand même.
- Faux négatifs : annulations "surprises" sur des profils très fermes (acompte total, non remboursable).

## 7. Schenyolla — Recommandation métier (3:30–4:10)
- Précision ~34% → **jamais d'action punitive** contre le client.
- Priorisation graduée en 4 niveaux : rien (<0,30) → e-mail J-7 (0,30–seuil) → contact personnalisé
  (seuil–0,60) → ajustement interne de planification (>0,60, invisible pour le client).

## 8. Toky — Conclusion (4:10–4:30)
- Modèle XGBoost, **F1 = 0,4814**, seuil 0,425.
- `submission.csv` généré et vérifié sur les 2000 réservations de test.
- Merci.

---

**Rappel pratique :** ouvrez `dashboard/index.html` (double-clic, pas de serveur nécessaire) et montrez la
section correspondante à l'écran pendant chaque partie.
