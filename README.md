## Objectif

Le projet produit deux versions du dataset `data/raw.csv` :

- `data/cleaned.csv` : version nettoyee techniquement ;
- `data/ethical_cleaned.csv` : version nettoyee puis reduite selon des choix ethiques.

L'idee est de corriger les problemes de qualite de donnees tout en limitant l'usage de variables sensibles dans un contexte proche d'une decision automatisee.

## Structure du dataset brut

Le jeu source contient `10000` lignes et `19` colonnes.

- Variables numeriques : `age`, `taille`, `poids`, `revenu_estime_mois`, `historique_credits`, `risque_personnel`, `score_credit`, `loyer_mensuel`, `montant_pret`
- Variables categorielles : `sexe`, `sport_licence`, `region`, `smoker`, `nationalité_francaise`, `situation_familiale`
- Variable ordinale : `niveau_etude`
- Variable temporelle : `date_creation_compte`
- Identifiants directs : `nom`, `prenom`

## Constats sur le dataset brut

Les principaux problemes observes sont les suivants :

- beaucoup de valeurs manquantes, surtout dans `score_credit`, `historique_credits`, `loyer_mensuel` et `situation_familiale` ;
- des valeurs incoherentes, notamment des `loyer_mensuel` negatifs ;
- des valeurs extremes sur plusieurs colonnes numeriques ;
- la presence de donnees personnelles ou sensibles (`nom`, `prenom`, `sexe`, `nationalité_francaise`, `smoker`, `taille`, `poids`).

## Pipeline de nettoyage technique

Le nettoyage applique dans `modules/cleaning.py` suit les etapes suivantes.

### 1. Standardisation des valeurs invalides

Les marqueurs de valeurs manquantes (`?`, `NA`, `N/A`, `null`, `None`) sont convertis en `NaN`.

Les valeurs negatives incoherentes sont aussi remplacees par `NaN` pour :

- `loyer_mensuel`
- `revenu_estime_mois`
- `montant_pret`
- `score_credit`

Choix retenu : transformer ces anomalies en valeurs manquantes permet de les traiter proprement dans une logique unique d'imputation.

### 2. Suppression des lignes les plus incompletes

Les `2%` de lignes les plus degradees sont supprimees.

Choix retenu : eviter d'imputer des lignes trop pauvres en information, tout en conservant l'essentiel du dataset.

### 3. Preparation avant imputation

Deux transformations sont appliquees avant l'imputation :

- `date_creation_compte` est convertie en `anciennete_compte_jours` ;
- `niveau_etude` est encodee selon un ordre logique :
  - `aucun` -> `0`
  - `bac` -> `1`
  - `bac+2` -> `2`
  - `bac+3` -> `3`
  - `bac+4` -> `4`
  - `master` -> `5`
  - `doctorat` -> `6`

Choix retenu : l'imputation fonctionne mieux sur des variables numeriques que sur des dates ou des niveaux textuels ordonnes.

### 4. Traitement des outliers

Les outliers sont detectes avec la methode IQR puis capes aux bornes inferieures et superieures.

Choix retenu : on reduit l'impact des valeurs extremes sans supprimer davantage de lignes.

### 5. Imputation des valeurs manquantes

Deux strategies sont utilisees selon le type de colonne :

- colonnes categorielles : `SimpleImputer(strategy="most_frequent")`
- colonnes numeriques : `KNNImputer`

Choix retenu :

- la modalite la plus frequente est simple et lisible pour les colonnes texte ;
- KNN permet une estimation plus proche des profils voisins pour les colonnes numeriques.

### 6. Post-traitement apres imputation

Certaines colonnes sont remises dans un format plus coherent :

- `historique_credits` est arrondie puis bornee entre `0` et `5` ;
- `niveau_etude` est arrondie, bornee, puis reconvertie en libelles ;
- `anciennete_compte_jours` est arrondie et forcee a rester positive.

Choix retenu : eviter des sorties absurdes comme des niveaux d'etude intermediaires ou des comptes d'anciennete negative.

## Sorties produites

### `data/cleaned.csv`

Ce fichier contient le dataset apres nettoyage technique complet.

- `9800` lignes ;
- plus aucune valeur manquante ;
- plus de loyers negatifs ;
- `date_creation_compte` remplacee par `anciennete_compte_jours`.

### `data/ethical_cleaned.csv`

Ce fichier est derive de `cleaned.csv` apres suppression de colonnes jugees trop sensibles ou inutiles pour un usage metier prudent :

- `nom`
- `prenom`
- `sexe`
- `taille`
- `poids`
- `nationalité_francaise`
- `smoker`

Choix retenu : appliquer une logique de minimisation des donnees et reduire le risque de discrimination ou de reidentification.
