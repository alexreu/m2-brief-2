## Partie 1 - Analyse exploratoire rapide

Ce premier passage sur le dataset sert a mesurer son etat avant nettoyage.

- Le jeu de donnees contient `10 000` lignes et `9` colonnes numeriques.
- Les colonnes les plus incompletes sont `score_credit` (`53.06%`), `historique_credits` (`52.93%`) et `loyer_mensuel` (`29.06%`).
- Aucune colonne ne depasse `80%` de valeurs manquantes, mais deux colonnes depassent `50%`, ce qui les rend deja fragiles pour la suite.
- `8 446` lignes ont au moins une valeur manquante. Les lignes les plus degradees montent a `33.33%` de valeurs manquantes, soit `3` champs vides sur `9`.
- Les boxplots et la detection par IQR montrent des valeurs atypiques sur `taille`, `poids`, `revenu_estime_mois` et `montant_pret`.
- Une anomalie evidente apparait sur `loyer_mensuel`, avec des valeurs negatives, peu coherentes metier.
- Les histogrammes confirment des distributions heterogenes selon les variables.
- La matrice de correlation montre peu de relations lineaires fortes. Le lien le plus visible reste celui entre `revenu_estime_mois` et `montant_pret`, mais il reste faible.

En clair, le dataset est exploitable, mais il demande un travail de nettoyage avant toute utilisation dans un modele.

## Partie 2 - Nettoyage

- Aucune colonne n'a ete supprimee pour cause de quasi-vide : aucune ne depasse `80%` de valeurs manquantes.
- Les `2%` de lignes les plus incompletes sont supprimees pour eviter d'imputer des observations trop degradees.
- Les valeurs manifestement incoherentes sont converties en `NaN`, par exemple les `loyer_mensuel`, `revenu_estime_mois`, `montant_pret` ou `score_credit` negatifs.
- Les outliers sont detectes avec la methode IQR puis capes aux bornes pour limiter leur impact sans perdre de lignes.
- L'imputation est hybride : `KNNImputer` pour les colonnes les plus incompletees (`>= 30%` de NaN), mediane pour le reste.

L'idee generale est simple : supprimer le minimum, corriger ce qui est incoherent.

## Partie 3 - Documentation du traitement et analyse statistique avant/apres

- Le nettoyage retire `200` lignes, soit `2%` du dataset, et fait passer le volume de `10 000` a `9 800` lignes.
- Les valeurs manquantes passent de `13 505` a `0`. Les lignes incompletes passent de `8 446` a `0`.
- Aucune colonne n'est supprimee : le seuil de quasi-vide (`80%`) n'est jamais atteint.
- Les valeurs incoherentes sont converties en `NaN`, puis traitees comme des manquants pour garder une logique unique de correction.
- Les outliers sont capes plutot que supprimes afin de conserver l'information tout en reduisant l'effet des valeurs extremes.
- Les colonnes tres incompletees sont imputees par KNN pour mieux preserver les profils individuels ; la mediane reste utilisee sur les colonnes moins touchees pour garder un traitement simple et robuste.

Sur le plan statistique, le nettoyage conserve les ordres de grandeur globaux, mais resserre les distributions.

- `taille` passe de `119.2 - 209.8` a `142.8 - 197.2`.
- `poids` passe de `10.5 - 145.2` a `29.05 - 111.05`.
- `revenu_estime_mois` passe de `500 - 6826` a `500 - 5731.12`.
- `montant_pret` passe de `500 - 53192.05` a `500 - 39817.46`.
- `loyer_mensuel` n'a plus de valeur negative apres correction, avec un minimum a `8.51` au lieu de `-395.25`.
- `score_credit` et `historique_credits` gardent une dispersion plus naturelle qu'avec une mediane seule, ce qui limite l'effet de concentration autour d'une valeur unique.

Le dataset final est plus propre, sans valeurs manquantes ni incoherences evidentes, tout en conservant mieux la variabilite des colonnes les plus degradees. C'est un compromis plus credible pour preparer les donnees avant la suite du projet.
