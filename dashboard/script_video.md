# Script vidéo — 3 à 5 minutes

## 0:00–0:30 — Intro et équipe (RAZAFIMBELO Toky Faniry)
"Bonjour, nous sommes l'équipe [nom], et nous allons présenter notre solution pour Atlantic Haven Hotels :
prédire l'annulation d'une réservation hôtelière." → présenter rapidement les 6 membres et leurs rôles
(EDA, baseline, modélisation, feature engineering, interprétation, soumission).

## 0:30–1:15 — Constats de l'EDA (RAVELONARIVO Fanantenana Mickaël)
- Cible déséquilibrée : 25,8% d'annulations → justifie le F1-score plutôt que l'accuracy.
- `agent_id` manquant à 42% n'est pas un problème : structurellement vide pour les canaux directs.
- Le taux d'annulation double au-delà d'un certain délai de réservation (relation non linéaire).
- Protocole de validation **temporel obligatoire** : split chronologique (train jusqu'au 2024-11-28,
  validation après) car le test est postérieur au train — une validation croisée aléatoire serait trompeuse.

## 1:15–1:45 — Baseline (IALISOA Iris Fifaliana)
"Nous avons d'abord construit une régression logistique avec tous les prétraitements appris uniquement sur
le train." → F1 = 0,4727 (seuil optimisé à 0,30). C'est le score de référence à battre.

## 1:45–2:45 — Modélisation et choix du modèle final (RAZAFIMBELO Toky Faniry)
- Comparaison de 3 modèles au seuil optimal : régression logistique (0,4727), Random Forest (0,4761),
  **XGBoost (0,4814)** — modèle retenu.
- Réglage par validation croisée temporelle (`TimeSeriesSplit`), pas de fuite train/test.
- Montrer le dashboard : tableau de comparaison + courbe seuil/F1 → expliquer le compromis
  précision/rappel au seuil 0,425 (rappel = 0,83, précision = 0,34).

## 2:45–3:15 — Feature engineering (RAMEFIARISON Fabio Fandresena)
"Nous avons créé des variables comme `personnes_par_chambre`, `prix_relatif_destination` et
`taux_annulation_passee`." → **résultat inattendu et honnête** : gain réel sur la régression logistique
(+0,0041 F1) mais aucun gain reproductible sur XGBoost (testé sur 5 graines), car les arbres reconstruisent
déjà ces interactions. Le modèle final reste XGBoost sans ces variables.

## 3:15–4:00 — Analyse d'erreurs et interprétation (RAKOTOARIMANANA Tojo Ny Aina)
- Variables les plus importantes : `type_acompte`, `tarif_remboursable`, `canal_reservation` — tous des
  leviers commerciaux actionnables par l'hôtel.
- Faux positifs : clients qui cumulent les signaux de risque (pas d'acompte, tarif remboursable) mais
  honorent leur réservation — l'intention réelle du client n'est pas dans les données.
- Faux négatifs : annulations « surprises » sur des profils très fermes (acompte total, non remboursable) —
  causes exogènes (imprévus) non observables.
- Montrer la matrice de confusion du dashboard.

## 4:00–4:40 — Recommandation métier (FANAMBIHARINDRAINY Schenyolla Anderssen)
"Avec une précision limitée (~34%), une action punitive toucherait trop de clients fidèles." →
présenter le tableau de priorisation graduée à 4 niveaux (dashboard) : parcours standard → e-mail de
confirmation → contact personnalisé → ajustement de la planification interne (jamais de sanction directe
au client).

## 4:40–5:00 — Conclusion
"Notre modèle XGBoost atteint un F1 de 0,4814 sur la validation temporelle, submission.csv généré et
vérifié sur les 2000 réservations de test. Merci de votre attention."

---

**Conseil pratique :** ouvrez `dashboard/index.html` dans un navigateur pendant l'enregistrement pour
montrer les graphiques à l'écran (double-clic sur le fichier, aucun serveur nécessaire).
