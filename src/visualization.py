"""
visualization.py
----------------
Fonctions de visualisation : sparsité, distributions, espace latent UMAP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import umap

plt.style.use('seaborn-v0_8-whitegrid')


def plot_sparsity(user_item_matrix: csr_matrix, sample_size: int = 50):
    """
    Heatmap (50x50) et scatter global de la sparsité de la matrice.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(
        user_item_matrix[:sample_size, :sample_size].toarray(),
        cmap='viridis', cbar=True,
        cbar_kws={'label': 'Note (Rating)'},
        ax=ax1
    )
    ax1.set_title(f'Zoom : Intensite des notes ({sample_size}x{sample_size} premiers)')
    ax1.set_xlabel('ID des Films')
    ax1.set_ylabel('ID des Utilisateurs')

    users_idx, movies_idx = user_item_matrix.nonzero()
    ax2.scatter(movies_idx, users_idx, alpha=0.3, s=0.05, color='royalblue')
    ax2.set_title('Densite globale : Repartition des interactions')
    ax2.set_xlabel('ID des Films')
    ax2.set_ylabel('ID des Utilisateurs')

    plt.suptitle('Analyse visuelle de la Matrice Utilisateur-Item', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_activity_distributions(df_ratings: pd.DataFrame):
    """
    Distributions lineaire et logarithmique du nombre de notes par utilisateur et par film.
    """
    ratings_per_user  = df_ratings.groupby('userId').size()
    ratings_per_movie = df_ratings.groupby('movieId').size()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for col, data, color, label in [
        (0, ratings_per_user,  'steelblue', 'Utilisateur'),
        (1, ratings_per_movie, 'coral',     'Film'),
    ]:
        for row, log in [(0, False), (1, True)]:
            axes[row, col].hist(data, bins=50, color=color, alpha=0.7,
                                edgecolor='black', log=log)
            axes[row, col].axvline(data.median(), color='red', linestyle='--',
                                   label=f'mediane: {data.median():.0f}')
            axes[row, col].legend()
            scale = 'Log Scale' if log else 'Lineaire'
            axes[row, col].set_title(f'Dist. des notes par {label} ({scale})')

    plt.tight_layout()
    return fig


def plot_svd_diagnostics(svd):
    """
    Valeurs singulieres et variance expliquee cumulee.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    ax1.plot(range(1, len(svd.singular_values_) + 1), svd.singular_values_, marker='o')
    ax1.set_title('Valeurs singulieres (TruncatedSVD)')
    ax1.set_xlabel('Indice de la composante')
    ax1.set_ylabel('Valeur singuliere')
    ax1.grid(True, alpha=0.3)

    cumvar = np.cumsum(svd.explained_variance_ratio_)
    ax2.plot(range(1, len(cumvar) + 1), cumvar, marker='o', color='darkgreen')
    ax2.axhline(0.8, color='red',    linestyle='--', label='80 %')
    ax2.axhline(0.9, color='orange', linestyle='--', label='90 %')
    ax2.set_title('Variance expliquee cumulee')
    ax2.set_xlabel('Nombre de composantes')
    ax2.set_ylabel('Variance cumulee')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_umap_space(item_factors: np.ndarray, movies_df: pd.DataFrame,
                    anchor_movie_ids: list = None):
    """
    Projection UMAP de l'espace latent SVD, avec mise en evidence optionnelle
    des films d'ancrage.
    """
    umap_model     = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    item_factors2d = umap_model.fit_transform(item_factors)

    latent_df = pd.DataFrame(item_factors2d, columns=['x', 'y'])
    latent_df['movieId'] = latent_df.index + 1

    movies_ext = movies_df.copy()
    movies_ext['primary_genre'] = movies_ext['genres'].str.split('|').str[0]
    movies_valid = movies_ext[movies_ext['movieId'] <= item_factors.shape[0]]

    latent_df = latent_df.merge(
        movies_valid[['movieId', 'title', 'primary_genre']], on='movieId', how='left'
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=latent_df, x='x', y='y', hue='primary_genre',
                    palette='tab20', alpha=0.5, s=15, legend=False, ax=ax)

    if anchor_movie_ids:
        anchors = latent_df[latent_df['movieId'].isin(anchor_movie_ids)]
        ax.scatter(anchors['x'], anchors['y'], color='red', s=120,
                   edgecolor='black', zorder=5, label='Films d\'ancrage')
        for _, row in anchors.iterrows():
            ax.text(row['x'] + 0.1, row['y'] + 0.1, row['title'],
                    fontsize=8, weight='bold')
        ax.legend()

    ax.set_title('Espace latent SVD — Projection UMAP')
    ax.set_xlabel('UMAP dimension 1')
    ax.set_ylabel('UMAP dimension 2')
    plt.tight_layout()
    return fig, latent_df
