"""
matrix_utils.py
---------------
Construction et manipulation de la matrice utilisateur-item sparse.
Filtrage et centrage par utilisateur.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def build_user_item_matrix(df_ratings: pd.DataFrame):
    """
    Construit une matrice utilisateur-item au format CSR a partir d'un DataFrame de ratings.

    Parameters
    ----------
    df_ratings : pd.DataFrame
        Colonnes requises : 'userId', 'movieId', 'rating'.

    Returns
    -------
    user_item_matrix : csr_matrix  (n_users x n_movies)
    n_users  : int
    n_movies : int
    """
    row_indices = df_ratings['userId'].values - 1
    col_indices = df_ratings['movieId'].values - 1
    ratings     = df_ratings['rating'].values
    n_users     = df_ratings['userId'].max()
    n_movies    = df_ratings['movieId'].max()

    user_item_matrix = csr_matrix(
        (ratings, (row_indices, col_indices)),
        shape=(n_users, n_movies)
    )

    sparsity = 1 - user_item_matrix.nnz / (n_users * n_movies)
    print(f"Matrice construite : {n_users} x {n_movies} | sparsité : {sparsity:.4%}")
    return user_item_matrix, n_users, n_movies


def filter_matrix(df_ratings: pd.DataFrame, min_ratings: int = 50):
    """
    Filtre les utilisateurs et les films ayant moins de `min_ratings` evaluations.
    """
    ratings_per_user  = df_ratings.groupby('userId').size()
    ratings_per_movie = df_ratings.groupby('movieId').size()

    valid_users  = ratings_per_user[ratings_per_user >= min_ratings].index
    valid_movies = ratings_per_movie[ratings_per_movie >= min_ratings].index

    filtered = df_ratings[
        df_ratings['userId'].isin(valid_users) &
        df_ratings['movieId'].isin(valid_movies)
    ]
    print(f"Apres filtrage (seuil={min_ratings}) : {len(valid_users)} users, {len(valid_movies)} films")
    return filtered


def center_by_user(user_item_matrix: csr_matrix):
    """
    Centre la matrice en soustrayant la note moyenne de chaque utilisateur.
    Corrige les biais systematiques de notation inter-utilisateurs.
    """
    matrix = user_item_matrix.copy().tocsr()
    n_users = matrix.shape[0]

    user_rating_sum   = np.array(matrix.sum(axis=1)).flatten()
    user_rating_count = np.diff(matrix.indptr)

    user_mean = np.zeros(n_users)
    mask = user_rating_count > 0
    user_mean[mask] = user_rating_sum[mask] / user_rating_count[mask]

    for i in range(n_users):
        s, e = matrix.indptr[i], matrix.indptr[i + 1]
        if s < e:
            matrix.data[s:e] -= user_mean[i]

    return matrix, user_mean
