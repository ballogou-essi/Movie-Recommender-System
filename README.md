# Movie Recommender System
### MovieLens 25M - Filtrage collaboratif, SVD & visualisation UMAP

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn) ![UMAP](https://img.shields.io/badge/UMAP-Dimensionality%20Reduction-9cf)

---

## Contexte

À partir de 25 millions d'évaluations de films, trois approches de recommandation sont implémentées et comparées. 

**L'objectif** : prédire les films qu'un utilisateur est susceptible d'apprécier parmi des milliers qu'il n'a pas encore vus.

---
👉 https://ballogou-essi.github.io/Movie-Recommendations/
---

## Approches comparées

| Méthode | Principe | Type |
|---|---|---|
| CF Item-based | Similarité cosinus entre films | Algèbre linéaire |
| CF User-based | Similarité cosinus entre utilisateurs | Algèbre linéaire |
| SVD tronquée (k=50) | Factorisation matricielle - espace latent | Machine Learning |

---

## Structure du projet

```
├── notebooks/
│   └── recommendation_system.ipynb  # Notebook principal
├── src/
│   ├── collaborative_filtering.py   # CF item-based & user-based
│   ├── svd_recommender.py           # Factorisation SVD
│   ├── matrix_utils.py              # Construction matrice sparse
│   └── visualization.py            # Graphiques & UMAP
├── figures/                         # Visualisations exportées
├── data/
│   └── README_data.md               # Instructions téléchargement MovieLens
├── requirements.txt
└── README.md
```

---

## Données

**Dataset :** [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) - GroupLens, University of Minnesota

| Dimension | Valeur |
|---|---|
| Interactions | 25 000 095 |
| Utilisateurs | 162 541 |
| Films | 209 171 |
| Sparsité de la matrice | 99,93 % |

> Les fichiers CSV ne sont pas versionnés (trop volumineux). Voir `data/README_data.md`.

---

## Points clés

- Matrice sparse au format **CSR** (SciPy) - impossible à stocker en dense (≈272 Go)
- Centrage par utilisateur avant SVD pour corriger les biais de notation
- Réduction dimensionnelle **UMAP** pour visualiser l'espace latent appris
- Analyse critique des limites : sparsité, biais de popularité, cold start

---

## Lancer le projet

```bash
pip install -r requirements.txt

jupyter notebook notebooks/recommendation_system.ipynb
```

---

## Stack technique

`Python` · `pandas` · `NumPy` · `SciPy` · `scikit-learn` · `UMAP` · `Matplotlib` · `Seaborn`
