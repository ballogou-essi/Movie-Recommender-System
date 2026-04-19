# Donnees - MovieLens 25M

## Source

Jeu de donnees publie par le GroupLens Research Lab, Universite du Minnesota.
Reference standard dans la litterature sur les systemes de recommandation.

- Page officielle  : https://grouplens.org/datasets/movielens/25m/
- Telechargement   : https://files.grouplens.org/datasets/movielens/ml-25m.zip
- Licence          : usage libre pour la recherche et l'education

## Fichiers utilises

| Fichier       | Description                                  | Taille |
|---------------|----------------------------------------------|--------|
| `movies.csv`  | movieId, titre, genres                       | ~3 Mo  |
| `ratings.csv` | userId, movieId, rating, timestamp           | ~640 Mo|

## Installation

```bash
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip -d data/
```

Adapter les chemins dans le notebook :

```python
df_movies  = pd.read_csv('data/ml-25m/movies.csv')
df_ratings = pd.read_csv('data/ml-25m/ratings.csv')
```

## Memoire requise

Le chargement complet de `ratings.csv` requiert environ 2 Go de RAM.
Prevoir au minimum 8 Go disponibles pour l'execution complete du notebook.

## Citation

Harper, F.M. & Konstan, J.A. (2015). The MovieLens Datasets: History and Context.
ACM Transactions on Interactive Intelligent Systems, 5(4), 1-19.
https://doi.org/10.1145/2827872
