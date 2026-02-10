"""
collaborative_filtering.py
--------------------------
Filtrage collaboratif item-based et user-based par similarite cosinus.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def get_similar_movies_item_based(
    item_user_matrix: csr_matrix,
    anchor_movie_id: int,
    movies_df: pd.DataFrame,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Retourne les top_k films les plus similaires a un film donne
    par similarite cosinus dans l'espace utilisateur (item-based CF).

    Parameters
    ----------
    item_user_matrix : csr_matrix  (n_movies x n_users)
    anchor_movie_id  : int         movieId du film de reference
    movies_df        : pd.DataFrame
    top_k            : int

    Returns
    -------
    pd.DataFrame avec colonnes : movieId, title, genres, similarity
    """
    anchor_idx    = anchor_movie_id - 1
    anchor_vector = item_user_matrix[anchor_idx]

    sims               = cosine_similarity(anchor_vector, item_user_matrix).flatten()
    sims[anchor_idx]   = -1  # exclusion du film lui-meme

    top_indices   = sims.argsort()[-top_k:][::-1]
    top_movie_ids = top_indices + 1

    results = movies_df[movies_df['movieId'].isin(top_movie_ids)][
        ['movieId', 'title', 'genres']
    ].copy()

    results['similarity'] = results['movieId'].apply(lambda mid: sims[mid - 1])
    return results.sort_values('similarity', ascending=False)


def get_user_based_recommendations(
    user_item_matrix: csr_matrix,
    fake_user_vector: csr_matrix,
    movies_df: pd.DataFrame,
    rated_movie_ids: set,
    top_k_users: int = 20,
    top_k_reco: int = 10
) -> pd.DataFrame:
    """
    Recommandations user-based : identifie les top_k_users voisins les plus proches
    et recommande les films bien notes par ces voisins mais non encore vus.

    Parameters
    ----------
    user_item_matrix  : csr_matrix (n_users x n_movies)
    fake_user_vector  : csr_matrix (1 x n_movies)
    movies_df         : pd.DataFrame
    rated_movie_ids   : set  movieIds deja notes par l'utilisateur cible
    top_k_users       : int  nombre de voisins
    top_k_reco        : int  nombre de recommandations

    Returns
    -------
    pd.DataFrame avec colonnes : movieId, title, genres, predicted_score
    """
    user_sims   = cosine_similarity(fake_user_vector, user_item_matrix).flatten()
    top_indices = user_sims.argsort()[-top_k_users:][::-1]

    rated_indices = set(mid - 1 for mid in rated_movie_ids)
    movie_scores  = {}

    for user_idx in top_indices:
        row = user_item_matrix[user_idx]
        for m_idx, rating in zip(row.indices, row.data):
            if m_idx in rated_indices:
                continue
            movie_scores.setdefault(m_idx, []).append(rating)

    mean_scores = {m: sum(r) / len(r) for m, r in movie_scores.items()}
    top_reco    = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_reco]

    reco_ids = [m + 1 for m, _ in top_reco]
    results  = movies_df[movies_df['movieId'].isin(reco_ids)][['movieId', 'title', 'genres']].copy()
    results['predicted_score'] = results['movieId'].apply(lambda mid: mean_scores[mid - 1])
    return results.sort_values('predicted_score', ascending=False)
