"""
svd_recommender.py
------------------
Factorisation matricielle par SVD tronquee.
Projection d'un nouvel utilisateur et prediction de scores.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def fit_svd(user_item_matrix_centered: csr_matrix, k: int = 50):
    """
    Ajuste une TruncatedSVD sur la matrice centree.

    Returns
    -------
    svd          : TruncatedSVD  modele ajuste
    user_factors : ndarray (n_users x k)
    item_factors : ndarray (n_movies x k)
    """
    svd          = TruncatedSVD(n_components=k, random_state=42)
    user_factors = svd.fit_transform(user_item_matrix_centered)
    item_factors = svd.components_.T  # (n_movies x k)
    return svd, user_factors, item_factors


def project_new_user(
    fake_ratings: dict,
    item_factors: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Projete un nouvel utilisateur dans l'espace latent par combinaison lineaire
    des facteurs items, ponderee par les notes centrees.

    Parameters
    ----------
    fake_ratings  : dict  {movieId: rating}
    item_factors  : ndarray (n_movies x k)
    k             : int  dimension de l'espace latent

    Returns
    -------
    fake_user_latent : ndarray (k,)
    """
    indices   = np.array([mid - 1 for mid in fake_ratings.keys()])
    ratings   = np.array(list(fake_ratings.values()), dtype=float)
    mean_r    = ratings.mean()
    centered  = ratings - mean_r

    latent = np.zeros(k)
    for m_idx, r in zip(indices, centered):
        latent += r * item_factors[m_idx]
    latent /= len(indices)
    return latent, mean_r


def predict_scores(
    fake_user_latent: np.ndarray,
    fake_user_mean: float,
    item_factors: np.ndarray,
    exclude_indices: set
) -> np.ndarray:
    """
    Calcule les scores predits pour l'ensemble du catalogue.
    """
    scores = item_factors @ fake_user_latent + fake_user_mean
    for idx in exclude_indices:
        scores[idx] = -np.inf
    return scores


def get_svd_recommendations(
    fake_ratings: dict,
    item_factors: np.ndarray,
    movies_df: pd.DataFrame,
    k: int = 50,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Recommandations SVD pour un utilisateur donne.
    """
    latent, mean_r = project_new_user(fake_ratings, item_factors, k)
    exclude        = set(mid - 1 for mid in fake_ratings.keys())
    scores         = predict_scores(latent, mean_r, item_factors, exclude)

    top_indices   = np.argsort(scores)[-top_k:][::-1]
    top_movie_ids = top_indices + 1

    results = movies_df[movies_df['movieId'].isin(top_movie_ids)][['movieId', 'title', 'genres']].copy()
    results['predicted_score'] = results['movieId'].apply(lambda mid: scores[mid - 1])
    return results.sort_values('predicted_score', ascending=False)


def get_similar_movies_svd(
    item_factors: np.ndarray,
    anchor_movie_id: int,
    movies_df: pd.DataFrame,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Voisins les plus proches d'un film dans l'espace latent SVD (similarite cosinus).
    """
    anchor_idx    = anchor_movie_id - 1
    anchor_vector = item_factors[anchor_idx].reshape(1, -1)

    sims               = cosine_similarity(anchor_vector, item_factors).flatten()
    sims[anchor_idx]   = -1

    top_indices   = sims.argsort()[-top_k:][::-1]
    top_movie_ids = top_indices + 1

    results = movies_df[movies_df['movieId'].isin(top_movie_ids)][['movieId', 'title', 'genres']].copy()
    results['similarity'] = results['movieId'].apply(lambda mid: sims[mid - 1])
    return results.sort_values('similarity', ascending=False)
