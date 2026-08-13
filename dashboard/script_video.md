# Script vidéo — texte à dire (3 à 5 minutes)

> Lisez directement ces répliques. Gardez `dashboard/index.html` ouvert à l'écran pendant l'enregistrement
> et montrez la section correspondante au fur et à mesure.

---

## 1. Intro et équipe — RAZAFIMBELO Toky Faniry (0:00–0:30)

> « Bonjour, nous sommes l'équipe d'Atlantic Haven Hotels pour l'examen final de Machine Learning et Data
> Science. Notre mission : prédire l'annulation d'une réservation hôtelière, sans pénaliser inutilement les
> clients qui vont finalement honorer leur séjour.
>
> Nous sommes six : Fanantenana s'est occupé de l'exploration des données, Iris de la baseline, moi-même de
> la modélisation et de l'intégration technique, Fabio du feature engineering, Tojo de l'interprétation, et
> Schenyolla de la soumission finale. Je laisse Fanantenana présenter nos premiers constats. »

---

## 2. Constats de l'EDA — RAVELONARIVO Fanantenana Mickaël (0:30–1:15)

> « Merci Toky. Première chose que nous avons observée : la cible est déséquilibrée, seulement 25,8% des
> réservations de notre jeu d'entraînement sont annulées. C'est important, parce que ça veut dire qu'un
> modèle qui prédirait toujours "pas d'annulation" aurait déjà 74% de bonnes réponses sans être utile — c'est
> pour ça que le sujet nous demande d'optimiser le F1-score, et pas l'accuracy.
>
> Ensuite, sur les valeurs manquantes : la variable agent_id est vide dans 42% des cas, mais ce n'est pas un
> problème de données — elle est structurellement absente pour les réservations directes, par téléphone ou
> au site de l'hôtel. Ce n'est pas à corriger, c'est une information déjà capturée par le canal de
> réservation.
>
> Et un point essentiel pour la suite : le taux d'annulation augmente nettement avec le délai de réservation
> — plus on réserve longtemps à l'avance, plus le risque d'annulation double. C'est pour ça que nous avons
> mis en place une validation temporelle stricte : nos données sont ordonnées dans le temps, et le jeu de
> test représente des réservations plus récentes que l'entraînement. Une validation croisée aléatoire aurait
> mélangé passé et futur, et donné un score trompeur. Nous avons donc découpé chronologiquement : les 6400
> premières réservations pour l'entraînement, les 1600 suivantes pour la validation. »

---

## 3. Baseline — IALISOA Iris Fifaliana (1:15–1:45)

> « De mon côté, j'ai construit la baseline obligatoire : une régression logistique, avec tous les
> prétraitements — imputation, encodage — appris uniquement sur les données d'entraînement, pour éviter
> toute fuite vers la validation.
>
> Cette baseline obtient un F1-score de 0,4727 au seuil optimal de 0,30. C'est le score de référence que
> tous nos modèles suivants devaient dépasser pour justifier leur complexité. »

---

## 4. Modélisation et choix du modèle final — RAZAFIMBELO Toky Faniry (1:45–2:45)

> « À partir de cette base, j'ai comparé trois familles de modèles sur le même jeu de validation : la
> régression logistique à 0,4727, un Random Forest à 0,4761, et XGBoost à 0,4814 de F1-score. J'ai réglé
> les hyperparamètres par validation croisée temporelle — donc toujours en respectant l'ordre chronologique,
> jamais un découpage aléatoire.
>
> C'est XGBoost que nous avons retenu comme modèle final. Vous voyez ici, sur notre tableau de bord, l'effet
> du seuil de décision : au seuil de 0,425 que nous avons choisi, on obtient un rappel de 0,83 — on détecte
> 83% des vraies annulations — mais une précision de seulement 0,34. C'est un choix assumé : sur un problème
> où le signal disponible est modéré, avec un ROC-AUC d'environ 0,65, privilégier le rappel permet de
> maximiser le F1-score demandé par le sujet. »

---

## 5. Feature engineering — RAMEFIARISON Fabio Fandresena (2:45–3:15)

> « J'ai créé plusieurs variables : le nombre de personnes par chambre, le prix relatif par rapport à la
> destination, et le taux d'annulation passé du client. Et notre résultat est honnête, pas forcément celui
> qu'on attendait : sur la régression logistique, ces variables apportent un gain réel et reproductible,
> +0,0041 de F1. Mais sur XGBoost, notre modèle final, ce même jeu de variables n'apporte aucun gain
> mesurable, vérifié sur cinq graines aléatoires différentes.
>
> L'explication est logique : un modèle à base d'arbres comme XGBoost peut déjà reconstruire ces rapports
> tout seul, à partir des variables brutes, via ses découpages successifs. Le gain du feature engineering
> est donc bien démontré expérimentalement, mais seulement sur le modèle linéaire — c'est pour ça que nous
> gardons XGBoost sans ces variables pour la suite. »

---

## 6. Interprétation et analyse d'erreurs — RAKOTOARIMANANA Tojo Ny Aina (3:15–4:00)

> « J'ai analysé quelles variables pèsent le plus dans les décisions du modèle : en tête, le type d'acompte,
> la remboursabilité du tarif, et le canal de réservation. Ce sont tous des leviers commerciaux que l'hôtel
> peut actionner directement, pas des caractéristiques subies par le client.
>
> Sur la matrice de confusion que vous voyez à l'écran : nos faux positifs sont des réservations qui
> cumulent tous les signaux de risque — pas d'acompte, tarif remboursable — mais que le client a quand même
> honorées. Le modèle n'a pas vraiment tort sur le profil, il n'a simplement pas accès à l'intention réelle
> du client.
>
> Nos faux négatifs, à l'inverse, sont des annulations surprises : des réservations avec acompte total versé
> et tarif non remboursable, qui semblaient très fermes, mais qui ont pourtant été annulées — probablement
> pour des raisons extérieures aux données, comme un imprévu personnel. »

---

## 7. Recommandation métier — FANAMBIHARINDRAINY Schenyolla Anderssen (4:00–4:40)

> « Avec une précision d'environ 34%, deux réservations signalées comme "à risque" sur trois seront en
> réalité honorées. On ne peut donc surtout pas prendre de mesure punitive contre le client.
>
> Nous proposons plutôt une action graduée selon quatre tranches de probabilité : en dessous de 0,30, aucune
> action particulière. Entre 0,30 et notre seuil, un simple e-mail de confirmation. Entre le seuil et 0,60,
> un contact personnalisé, par exemple une offre de service additionnel. Et au-dessus de 0,60, on n'agit plus
> sur le client mais sur la planification interne de l'hôtel — un ajustement mesuré, invisible pour le
> client. Le principe est simple : une action qui coûte peu si elle est déclenchée à tort, mais qui protège
> l'hôtel dans les cas les plus à risque. »

---

## 8. Conclusion — RAZAFIMBELO Toky Faniry (4:40–5:00)

> « Pour résumer : notre modèle XGBoost atteint un F1-score de 0,4814 sur la validation temporelle, avec un
> seuil de décision de 0,425. Notre fichier submission.csv a été généré et vérifié sur les 2000 réservations
> du jeu de test, conformément à toutes les exigences du sujet. Merci de votre attention. »

---

**Rappel pratique :** ouvrez `dashboard/index.html` dans un navigateur (double-clic, aucun serveur requis)
et montrez la section correspondante à l'écran au fil du script.
