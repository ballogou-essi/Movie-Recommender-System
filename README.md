# Systeme de recommandation de films — Filtrage collaboratif et factorisation matricielle

> Projet de fin de module · Master 2 Machine Learning · Universite de Strasbourg  
> Dataset : MovieLens 25M · Algorithmes : CF item-based, CF user-based, SVD + UMAP

---
https://ballogou-essi.github.io/Movie-Recommendations/
---
## Table des matieres

1. [Contexte et problematique](#1-contexte-et-problematique)
2. [Description des donnees](#2-description-des-donnees)
3. [Approche methodologique](#3-approche-methodologique)
4. [Modeles et algorithmes](#4-modeles-et-algorithmes)
5. [Resultats](#5-resultats)
6. [Discussion critique](#6-discussion-critique)
7. [Axes d'amelioration](#7-axes-damelioration)
8. [Conclusion](#8-conclusion)
9. [Structure du depot](#9-structure-du-depot)
10. [Reproductibilite](#10-reproductibilite)

---

## 1. Contexte et problematique

### Motivation

La recommandation de contenu est l'un des problemes les mieux documentes en apprentissage automatique applique. Sa difficulte ne reside pas dans l'absence de donnees — au contraire — mais dans la nature structurellement lacunaire de ces donnees : la plupart des utilisateurs n'ont interagi qu'avec une infime partie du catalogue disponible. Le probleme central est donc celui de la **prediction de preferences implicites** a partir d'observations partielles.

Ce projet s'attaque precisement a ce probleme dans le contexte du cinema, en utilisant le jeu de donnees MovieLens 25M, reference academique incontournable dans le domaine. L'objectif est double : construire un systeme fonctionnel de recommandation de films, et comparer de maniere argumentee plusieurs approches algorithmiques, du plus simple au plus sophistique.

### Problematique

Etant donne un utilisateur ayant evalue un sous-ensemble de films, comment predire les films qu'il est susceptible d'apprecier parmi les milliers qu'il n'a pas encore vus ?

Deux angles d'attaque sont explores :

- **L'approche item-based** : recommander des films structurellement proches de ceux deja apprecies.
- **L'approche user-based** : identifier des utilisateurs au profil similaire et propager leurs preferences.
- **La factorisation matricielle (SVD)** : projeter utilisateurs et films dans un espace latent commun, et exploiter cette geometrie pour predire des notes manquantes.

---

## 2. Description des donnees

### Source

Les donnees proviennent du jeu de donnees **MovieLens 25M**, publie et maintenu par le GroupLens Research Lab de l'Universite du Minnesota. Il s'agit d'une reference standard dans la litterature sur les systemes de recommandation, utilisee dans de nombreux travaux academiques et benchmarks industriels.

### Structure

Le projet mobilise deux fichiers CSV :

| Fichier | Contenu | Taille |
|---|---|---|
| `movies.csv` | Identifiant, titre, genres de chaque film | ~62 000 films |
| `ratings.csv` | Triplets (userId, movieId, rating) + timestamp | 25 000 095 interactions |

```
# Apercu de movies.csv
   movieId                               title                                        genres
0        1                    Toy Story (1995)   Adventure|Animation|Children|Comedy|Fantasy
1        2                      Jumanji (1995)                    Adventure|Children|Fantasy
2        3             Grumpier Old Men (1995)                                Comedy|Romance

# Apercu de ratings.csv
   userId  movieId  rating   timestamp
0       1      296     5.0  1147880044
1       1      306     3.5  1147868817
2       1      307     5.0  1147868828
```

### Volumetrie et dimensions

La matrice utilisateur-item brute presente les caracteristiques suivantes :

| Dimension | Valeur |
|---|---|
| Nombre d'utilisateurs | 162 541 |
| Nombre de films (index max) | 209 171 |
| Nombre d'interactions | 25 000 095 |
| Taille theorique de la matrice | 33 998 863 511 cellules |
| **Taux de sparsité** | **99,93 %** |

Ce taux de sparsité de 99,93 % est la donnee structurelle la plus importante du projet. Elle signifie que sur l'ensemble des paires (utilisateur, film) concevables, moins d'une sur mille a effectivement donne lieu a une notation. Tout l'enjeu de la modelisation est de combler intelligemment ces vides.

### Distribution des interactions

L'analyse exploratoire revele une distribution fortement asymetrique, caracteristique des systemes de recommandation reels :

**Par utilisateur :**
- Moyenne : 153,8 notes par utilisateur
- Mediane : 71,0 notes
- Maximum : 32 202 notes
- Les 20 % d'utilisateurs les plus actifs representent la quasi-totalite des interactions

**Par film :**
- Moyenne : 423,4 notes par film
- Mediane : seulement 6,0 notes
- Maximum : 81 491 notes (films tres populaires)
- La grande majorite des films sont extremement peu evalues

Cette double asymetrie — quelques utilisateurs tres actifs, quelques films tres populaires — est typique d'une distribution en loi de puissance. Elle implique que les modeles seront naturellement plus performants sur les profils actifs et les films populaires, et qu'une attention particuliere doit etre portee aux cas limites.

### Filtrage et preparation

Pour eviter les biais dus aux utilisateurs et aux films peu representes, un seuil minimal de 50 evaluations est applique :

```python
filtered_users = ratings_per_user[ratings_per_user >= 50].index
filtered_movies = ratings_per_movie[ratings_per_movie >= 50].index
```

Apres filtrage :

| Dimension | Avant filtrage | Apres filtrage |
|---|---|---|
| Utilisateurs | 162 541 | 102 492 |
| Films | 209 171 | 13 176 |
| Sparsité | 99,93 % | 98,31 % |

Le filtrage reduit significativement la dimensionnalite tout en preservant la quasi-totalite de la richesse informationnelle, et ameliore la qualite des vecteurs de representation utilisateur et item.

---

## 3. Approche methodologique

### Choix du paradigme : filtrage collaboratif

Trois grands paradigmes existent en recommandation :

1. **Le filtrage base sur le contenu** (content-based) : exploiter les attributs intrinseques des items (genre, realisateur, acteurs, synopsis).
2. **Le filtrage collaboratif** (collaborative filtering, CF) : exploiter uniquement les interactions entre utilisateurs et items, sans recours aux attributs.
3. **Les methodes hybrides** : combiner les deux approches.

Le choix ici se porte sur le **filtrage collaboratif pur**, pour deux raisons principales. D'une part, MovieLens ne fournit pas de metadonnees riches (pas de synopsis, pas de casting detaille), ce qui limite l'interet d'une approche content-based. D'autre part, le CF est algorithmiquement plus riche et plus interessant a evaluer comparativement, car il couvre un spectre de complexite allant de la simple similarite cosinus jusqu'a la factorisation matricielle.

### Pipeline general

```
Donnees brutes (ratings.csv)
        |
        v
[1. Construction de la matrice utilisateur-item (sparse CSR)]
        |
        v
[2. Centrage par utilisateur (normalisation des biais)]
        |
     /-----\
    /       \
   v         v
[3a. CF item-based]    [3b. CF user-based]    [3c. SVD]
cosine similarity      cosine similarity       TruncatedSVD (k=50)
    |                       |                      |
    v                       v                      v
[Recommandations top-K pour les films ancres et le fake user]
        |
        v
[4. Visualisation de l'espace latent (UMAP 2D)]
        |
        v
[5. Analyse des voisins dans l'espace latent]
```

### Representation en matrice sparse

Un choix technique fondamental est l'utilisation du format **CSR (Compressed Sparse Row)** de SciPy. Stocker la matrice complete en memoire dense serait impossible (34 milliards de cellules en float64 = environ 272 Go). Le format CSR ne stocke que les entrees non nulles, reduisant l'empreinte memoire a quelques centaines de megaoctets.

```python
user_item_matrix = csr_matrix(
    (ratings, (row_indices, col_indices)),
    shape=(n_users, n_movies)
)
```

### Films d'ancrage

Pour evaluer qualitativement les recommandations, trois films sont choisis comme points d'ancrage representant des profils stylistiquement distincts :

| Film | movieId | Genre dominant |
|---|---|---|
| The Greatest Showman (2017) | 180 985 | Musical / Drama |
| The Lion King (2019) | 203 222 | Animation / Adventure |
| I Want to Eat Your Pancreas | 198 611 | Animation / Drama japonais |

Ce choix deliberement contraste permet d'observer le comportement des algorithmes dans des configurations tres differentes : un film grand public a forte audience, un blockbuster d'animation recente, et un film d'animation japonais de niche tres peu note dans le corpus.

---

## 4. Modeles et algorithmes

### 4.1 Filtrage collaboratif item-based (cosine similarity)

**Principe :** Deux films sont consideres comme similaires si les utilisateurs qui ont note l'un ont tendance a avoir aussi note l'autre, et dans les memes proportions. La proximite est mesuree par la **similarite cosinus** entre les vecteurs d'evaluations des items dans l'espace utilisateur.

Pour deux films i et j, la similarite s'ecrit :

```
sim(i, j) = cos(v_i, v_j) = (v_i . v_j) / (||v_i|| * ||v_j||)
```

ou `v_i` est le vecteur des notes recues par le film i de la part de l'ensemble des utilisateurs.

**Interet par rapport a d'autres metriques :** La similarite cosinus est invariante a l'echelle, ce qui la rend robuste au fait que certains utilisateurs notent systematiquement plus haut ou plus bas que la moyenne. Elle ne mesure pas la magnitude des vecteurs mais leur orientation dans l'espace, ce qui est precis ment ce que l'on cherche ici.

**Implementation :**

```python
# Transposition : items en lignes, utilisateurs en colonnes
item_user_matrix = user_item_matrix.T  # shape: movies x users

def get_top_similar_movies_item_based(item_user_matrix, anchor_movie_id, movies_df, top_k=10):
    anchor_idx = anchor_movie_id - 1
    anchor_vector = item_user_matrix[anchor_idx]

    # Similarite cosinus entre le film ancre et tous les autres
    similarities = cosine_similarity(anchor_vector, item_user_matrix).flatten()
    similarities[anchor_idx] = -1  # exclusion du film lui-meme

    top_indices = similarities.argsort()[-top_k:][::-1]
    # ...
```

### 4.2 Filtrage collaboratif user-based

**Principe :** Plutot que de comparer des items, on compare des utilisateurs. Pour un utilisateur cible, on identifie les k utilisateurs les plus proches (en termes de profil de notation), puis on recommande les films bien notes par ces voisins et non encore vus par l'utilisateur cible.

Ce paradigme repose sur l'hypothese de **coherence comportementale** : des utilisateurs ayant aime les memes films dans le passe auront tendance a avoir des gouts similaires a l'avenir.

Pour tester cette approche, un utilisateur synthetique ("fake user") est construit manuellement avec un profil de gouts connu :

```python
fake_user_ratings = {
    180985: 5.0,  # The Greatest Showman
    203222: 4.5,  # The Lion King (2019)
    198611: 5.0,  # I Want to Eat Your Pancreas
    5690: 4.0,    # Grave of the Fireflies
    120258: 4.5,  # Shaka Zulu
    3404: 4.5,    # Titanic
}
```

Ce profil est ensuite projete comme un vecteur dans l'espace de la matrice utilisateur-item, et la similarite cosinus est calculee avec l'ensemble des utilisateurs reels.

### 4.3 Factorisation matricielle par SVD tronquee

**Principe :** La decomposition en valeurs singulieres tronquee (Truncated SVD) decomposes la matrice utilisateur-item R de dimension (m x n) en :

```
R ≈ U * Sigma * V^T
```

ou U (m x k) encode les utilisateurs dans l'espace latent, Sigma (k x k) contient les valeurs singulieres, et V^T (k x n) encode les items. Le parametre k (ici k=50) controle la dimensionnalite de l'espace latent.

L'intuition est que les k plus grandes valeurs singulieres capturent les "dimensions semantiques" les plus importantes du comportement collectif (par exemple : un axe "cinema grand public vs cinema d'auteur", un axe "preference pour les films recents vs les classiques", etc.).

**Centrage prealable :** Avant la decomposition, les notes sont centrees par utilisateur pour corriger les biais systematiques de notation :

```python
for user_idx in range(n_users):
    start = user_item_matrix_centered.indptr[user_idx]
    end   = user_item_matrix_centered.indptr[user_idx + 1]
    if start < end:
        user_item_matrix_centered.data[start:end] -= user_mean_rating[user_idx]
```

Ce centrage est crucial : sans lui, les dimensions dominantes de la decomposition refleteraient principalement les differences de niveau moyen entre utilisateurs plutot que leurs preferences relatives.

**Projection d'un nouvel utilisateur :** Pour projeter le fake user dans l'espace latent sans reinitialiser le modele :

```python
fake_user_latent = np.zeros(k)
for m_idx, r in zip(fake_movie_indices, fake_ratings_centered):
    fake_user_latent += r * item_factors[m_idx]
fake_user_latent /= len(fake_movie_indices)
```

Il s'agit d'une projection par combinaison lineaire des facteurs items, ponderee par les notes centrees. C'est une approximation valide dans le cadre de la SVD.

### 4.4 Reduction dimensionnelle et visualisation (UMAP)

Apres la factorisation SVD, les items existent dans un espace a 50 dimensions. Pour visualiser la geometrie de cet espace, une reduction supplementaire a 2 dimensions est effectuee par **UMAP (Uniform Manifold Approximation and Projection)**.

UMAP est prefere a t-SNE pour deux raisons : il preserve mieux la structure globale de l'espace latent (pas seulement les voisinages locaux), et il est nettement plus rapide sur de grandes matrices.

```python
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
item_factors_2d = umap_model.fit_transform(item_factors)
```

---

## 5. Resultats

### 5.1 Sparsité et structure de la matrice

La visualisation de la matrice sparse revele une structure caracteristique en deux zones :

- **La heatmap (50x50 premiers utilisateurs/films)** montre que la plupart des cellules sont vides, et que les evaluations existantes sont distribuees de facon heterogene. Certains utilisateurs ont note de nombreux films dans la fenetre observee, d'autres aucun.
- **Le scatter plot global** confirme la sparsité extreme (99,93 %) : les interactions se concentrent sur un sous-ensemble dense de la matrice, correspondant aux films populaires et aux utilisateurs actifs.

Cette structure implique que les algorithmes de CF souffriront necessairement d'un **probleme de demarrage a froid** pour les utilisateurs peu actifs et les films peu notes.

### 5.2 Recommandations item-based

**Pour The Greatest Showman (2017) :**

| Titre | Genres | Similarite |
|---|---|---|
| Jumanji: Welcome to the Jungle (2017) | Action / Adventure | 0.314 |
| Beauty and the Beast (2017) | Fantasy / Romance | 0.286 |
| Black Panther (2017) | Action / Adventure | 0.277 |
| Wonder (2017) | Drama | 0.272 |
| Ready Player One | Action / Sci-Fi | 0.271 |

Les scores de similarite cosinus avoisinent 0.26-0.31. Ces valeurs apparemment modestes sont normales dans un espace sparse de grande dimension : la similarite est mesuree sur un vecteur de 160 000 dimensions dont la quasi-totalite est nulle. L'important est l'ordre de grandeur relatif entre les films recommandes.

Les recommandations sont cohérentes : ce sont des films sortis en 2017-2018, grand public, a fort potentiel commercial. La proximite temporelle est un biais notable de la methode item-based dans ce contexte.

**Pour The Lion King (2019) :**

| Titre | Genres | Similarite |
|---|---|---|
| Aladdin (2019) | Adventure / Fantasy | 0.407 |
| Toy Story 4 (2019) | Animation / Children | 0.306 |
| Dumbo (2019) | Adventure / Children | 0.258 |
| Dark Phoenix (2019) | Action / Sci-Fi | 0.235 |
| Captain Marvel (2018) | Action / Adventure | 0.234 |

La similarite maximale (0.407 pour Aladdin) est sensiblement plus elevee. Aladdin est sorti la meme annee, appartient au meme studio (Disney live-action remakes), et partage une audience cible quasi identique. Ce resultat demontre que l'item-based CF capture bien les co-occurrences de visionnage dans des niches thematiques ou temporelles precises.

**Pour I Want to Eat Your Pancreas :**

| Titre | Genres | Similarite |
|---|---|---|
| The Last Trick (1964) | Animation | 0.524 |
| Transfert per camera verso Virulentia (1967) | Documentary | 0.453 |
| Invisible Ink (1921) | Animation / Comedy | 0.453 |
| Nostalgia (2018) | Drama | 0.452 |

Ce cas est le plus instructif. Les scores de similarite sont plus eleves (0.45-0.52), mais les films recommandes sont des titres tres peu connus, sans rapport thematique apparent avec le film d'ancrage. Ce phenomene s'explique par la **faible densite d'interactions** : "I Want to Eat Your Pancreas" est un film de niche peu note dans MovieLens (public occidental). Les quelques utilisateurs qui l'ont note ont des profils d'eclectiques, ce qui cree des co-occurrences artificielles avec des films tout aussi marginaux. C'est une illustration concrete de la **degradation du CF en regime sparse**.

### 5.3 Recommandations user-based (fake user)

| Titre | Genres | Score predit |
|---|---|---|
| Sudden Death (1995) | Action | 5.0 |
| Before the Rain (1994) | Drama / War | 5.0 |
| Life Is Beautiful (1997) | Comedy / Drama | 5.0 |
| Anatomy of a Murder (1959) | Drama / Mystery | 5.0 |
| The Notebook (2004) | Drama / Romance | 5.0 |

L'ensemble des recommandations user-based obtient un score predit de 5.0, ce qui constitue une anomalie methodologique a analyser.

Ce score de 5.0 resulte mecaniquement du fait que les voisins identifies ont, eux-memes, une tendance marquee a donner des notes maximales aux films qu'ils ont vu. Le score predit est la moyenne des notes des voisins, non une prediction probabiliste. En l'absence de regularisation ou de ponderation par la similarite, ce resultat ne discrimine pas les films recommandes entre eux : tous semblent egalement "parfaits". Ce comportement est typique d'un CF user-based naive sur des profils d'utilisateurs intensifs qui ont tendance a noter tres positivement.

### 5.4 Absence de recouvrement entre les deux approches

Un resultat particulierement revelateur est l'absence totale de recouvrement entre les recommandations item-based et user-based :

```
Films recommandes par LES DEUX methodes : aucun
Films recommandes uniquement par item-based : 29 films
Films recommandes uniquement par user-based : 10 films
```

Ce resultat n'est pas surprenant, mais il illustre de facon concrete que les deux paradigmes operent sur des criteres fondamentalement differents. L'item-based CF est ancre dans le contexte des films d'entree (films de 2016-2019, meme audience), tandis que le user-based CF reflète les preferences generales des voisins identifies, sans contrainte temporelle ni thematique.

### 5.5 Recommandations SVD

| Titre | Genres | Score predit |
|---|---|---|
| Blair Witch Project (1999) | Drama / Horror | 4.501 |
| Babe (1995) | Children / Drama | 4.501 |
| Independence Day (1996) | Action / Sci-Fi | 4.500 |
| Raiders of the Lost Ark | Action / Adventure | 4.500 |
| Twelve Monkeys (1995) | Mystery / Sci-Fi | 4.500 |
| Apollo 13 (1995) | Adventure / Drama | 4.500 |
| The Hangover (2009) | Comedy / Crime | 4.500 |
| American Beauty (1999) | Drama / Romance | 4.500 |

Le SVD produit des recommandations sensiblement differentes des deux approches precedentes : les scores sont moins homogenes (meme si restes tres proches autour de 4.50), et les films recommandes sont des classiques bien notes, representatifs d'un large consensus cinematographique. La methode recommande principalement des films populaires des annees 1990-2000 ayant accumule un grand nombre d'evaluations.

Ce resultat revele un biais inherent au SVD avec un nombre reduit de composantes : les dimensions latentes capturent en priorite la structure des films tres populaires. Un fake user ayant aime des films recents et de niche se voit recommander des classiques universellement apprecies, ce qui traduit une **regression vers la moyenne** caracteristique des factorisations avec peu de composantes.

### 5.6 Variance expliquee par le SVD

L'analyse de la variance expliquee cumulee par les 50 composantes latentes permet d'evaluer la qualite de la compression :

- Les premieres composantes capturent une part disproportionnee de la variance totale (decroissance rapide des valeurs singulieres).
- Le coude de la courbe de variance expliquee indique que les 10-15 premieres dimensions latentes concentrent l'essentiel de l'information structurelle.
- Les dimensions superieures capturent des gouts de plus en plus specifiques et marginaux.

Ce profil est coherent avec la theorie : dans un systeme de recommandation, quelques "megafacteurs" (popularite, genre dominant, epoque) expliquent une fraction importante des variations de notation, tandis que les preferences fines sont encodees dans les dimensions subsequentes.

### 5.7 Analyse de l'espace latent (UMAP)

La projection UMAP de l'espace latent SVD (50 dimensions -> 2 dimensions) revele plusieurs structures interessantes.

**Vue globale :** La projection fait apparaitre des clusters distincts, ce qui confirme que l'espace latent appris par la SVD possede une structure geometrique significative. Les films ne sont pas distribues aleatoirement : des zones de l'espace correspondent a des profils de visionnage coherents.

**Position des films d'ancrage :** Les trois films d'ancrage sont situes dans des regions bien separees de l'espace. The Greatest Showman et The Lion King occupent des zones denses (films populaires avec de nombreux voisins proches), tandis que "I Want to Eat Your Pancreas" se situe en peripherie.

**Voisins dans l'espace 2D de The Greatest Showman :**

| Titre | Genre | Distance |
|---|---|---|
| Clay Pigeons (1998) | Crime | 0.055 |
| Heidi Fleiss: Hollywood Madam (1995) | Documentary | 0.072 |
| Green Zone (2010) | Action | 0.079 |
| Torn Curtain (1966) | Thriller | 0.094 |

**Voisins dans l'espace 2D de I Want to Eat Your Pancreas :**

| Titre | Genre | Distance |
|---|---|---|
| Hooligan Sparrow (2016) | Documentary | 0.015 |
| The Agha (1985) | (no genres) | 0.015 |
| War and Peace (1965) | (no genres) | 0.015 |

La forte coherence entre les voisins de "I Want to Eat Your Pancreas" (tres faibles distances, films peu connus) suggere que le film se trouve dans une zone de l'espace latent ou les films "orphelins" (peu de notes, profil de visionnage tres specifique) se regroupent par defaut.

**Dispersion par genre dans l'espace latent :**

| Genre | Variance moyenne (2D) |
|---|---|
| Documentary | 10.21 |
| Horror | 14.39 |
| Romance | 16.72 |

Les documentaires forment les clusters les plus compacts, ce qui suggere que ce genre est associe a des comportements de visionnage particulierement coherents et specifiques. A l'inverse, le genre Romance montre la plus grande dispersion : les films romantiques sont vus par des publics tres heterogenes, ce qui les disperse dans l'espace latent.

---

## 6. Discussion critique

### Ce que montrent reellement les resultats

Les trois methodes produisent des recommandations de nature qualitativement differente, et cette divergence est elle-meme informative.

L'item-based CF se comporte de facon previsible sur les films populaires (co-occurrences temporelles et generiques fortes) mais echoue sur les films de niche. Le user-based CF capture une logique de "profil general" plutot qu'une logique de "film specifique", ce qui explique les recommandations de classiques tous genres confondus. La SVD, quant a elle, converge vers un ensemble de films populaires et bien etablis, refletant les structures dominantes de la matrice.

### Limites methodologiques

**La sparsité :** Meme apres filtrage, la matrice reste sparse a 98,31 %. Cela signifie que la majorite des comparaisons item-item ou user-user reposent sur un chevauchement tres faible de notes communes, ce qui fragilise la robustesse des scores de similarite.

**Le biais de popularite :** Les trois methodes ont tendance a sur-representer les films populaires. Un film qui a recu 80 000 notes occupera naturellement une position centrale dans l'espace de representation, au detriment des films de niche qui peuvent etre tout aussi pertinents pour un utilisateur donne.

**Le biais temporel :** L'item-based CF produit frequemment des recommandations de films sortis la meme annee que le film d'ancrage. Ce n'est pas necessairement une limite du modele, mais il convient de le signaler comme un artefact : les co-occurrences de notation sont correlees avec la co-disponibilite temporelle des films sur les plateformes.

**L'absence d'evaluation quantitative rigoureuse :** Ce projet ne dispose pas d'un protocole d'evaluation standard (pas de split train/test, pas de calcul de RMSE, Precision@K ou nDCG). Les jugements de qualite sont donc qualitatifs, bases sur la coherence semantique des recommandations. Cette limitation est inherente a la conception du projet mais devrait etre adressee dans une version ulterieure.

**Le probleme du fake user :** Construire un profil utilisateur synthetique pour evaluer les recommandations user-based est une approche heuristique. En l'absence d'un vrai utilisateur cible avec des preferences connues et validees, il est impossible de mesurer objectivement si les recommandations sont pertinentes.

### Robustesse du modele

La SVD avec k=50 est relativement robuste : le choix de k influe sur la qualite des recommandations mais la methode reste fonctionnelle dans une large plage de valeurs. En revanche, les methodes de CF non parametre (item-based, user-based) sont plus sensibles aux parametres de filtrage (seuil minimal de notes par utilisateur / film).

### Cas ou le modele echoue

- **Films recents peu notes :** Tout film ajoute au catalogue sans historique d'interactions suffisant sera mal represente dans tous les modeles.
- **Utilisateurs aux gouts tres specifiques :** Un utilisateur ayant aime exclusivement des films de niche n'aura que peu de voisins credibles dans le corpus.
- **Films multigenres :** Un film appartenant a plusieurs genres peut se retrouver dans une zone intermediaire de l'espace latent, eloigne de tous les clusters, et donc difficile a utiliser comme point d'ancrage.

---

## 7. Axes d'amelioration

### Ameliorations algorithmiques

**Evaluation rigoureuse :** Implementer un protocole d'evaluation standard avec split temporel (les notes anterieures a une date servent a l'apprentissage, les notes posterieures a l'evaluation). Calculer RMSE pour les methodes de prediction de notes, et Precision@K / Recall@K / nDCG@K pour les methodes de ranking.

**Ponderation de la similarite :** Dans le CF user-based, ponderer le score predit par la similarite avec chaque voisin, plutot que de calculer une moyenne brute. Cela eviterait la convergence artificielle vers des scores de 5.0.

**Methodes de factorisation plus avancees :** Essayer ALS (Alternating Least Squares) ou BPR (Bayesian Personalized Ranking), tous deux mieux adaptes aux matrices implicites. BPR en particulier est plus adequat lorsque les donnees d'interaction sont considerees comme des signaux d'interet binaires plutot que comme des notes continues.

**Regularisation L2 :** Ajouter une regularisation dans la phase de factorisation pour limiter l'overfitting sur les films tres populaires.

**Approche hybride :** Combiner le SVD avec des attributs de contenu (genre, annee, popularite) dans un modele de type LightFM pour corriger le biais de demarrage a froid.

### Ameliorations data

**Normalisation des notes :** En plus du centrage par utilisateur, normaliser les notes par item (centrage global) peut ameliorer la qualite de la factorisation.

**Enrichissement du corpus :** Integrer les tags fournis dans MovieLens 25M (fichier `tags.csv`) comme signal semantique supplementaire pour les films de niche.

**Prise en compte de la temporalite :** Ponderer les evaluations recentes plus fortement que les anciennes. Les preferences evoluent dans le temps, et une note donnee il y a 15 ans est moins informative qu'une note recente.

### Pistes d'industrialisation

Un passage a l'echelle industriel necessiterait l'adoption d'un systeme de mise a jour incrementale (online learning) plutot que de reinitialiser le modele a chaque nouveau batch de donnees. Des frameworks comme ALS distribue via Spark MLlib ou des solutions dedicees comme LensKit permettraient de scaler a des corpus de plusieurs centaines de millions d'interactions.

---

## 8. Conclusion

Ce projet construit et compare trois approches classiques de la recommandation de films sur le jeu de donnees MovieLens 25M. Les principaux enseignements sont les suivants.

Premierement, les methodes de filtrage collaboratif produisent des resultats qualitativement coherents sur les films populaires, mais montrent leurs limites sur les films de niche, victimes de la sparsité structurelle du probleme. Deuxiemement, les paradigmes item-based et user-based, bien que tous deux fondes sur la similarite cosinus, capturent des logiques de recommandation distinctes et produisent des recommandations sans recouvrement, ce qui confirme leur complementarite plutot que leur substituabilite. Troisiemement, la SVD apporte une couche d'interpretation supplementaire via l'espace latent : elle permet non seulement de recommander des films, mais aussi de visualiser et de quantifier la structure geometrique des preferences cinematographiques collectives.

Sur le plan methodologique, ce projet met en evidence plusieurs defis recurrents des systemes de recommandation reels : la gestion de la sparsité extreme, le biais de popularite, l'absence de metriques d'evaluation objectives, et le probleme du demarrage a froid. Ces defis ne sont pas specifiques au cinema ; ils se retrouvent dans tous les domaines ou les systemes de recommandation sont deployes.

En termes de competences, ce projet mobilise la maitrise du traitement de donnees sparse a grande echelle (25 millions d'interactions), l'implementation d'algorithmes de CF from scratch, la decomposition matricielle, la reduction dimensionnelle non lineaire (UMAP), et l'analyse critique des resultats experimentaux.

---

## 9. Structure du depot

```
recommendation-system/
|
|-- README.md                          # Ce document
|
|-- notebooks/
|   |-- recommendation_system.ipynb   # Notebook complet avec outputs
|
|-- data/
|   |-- README_data.md                # Instructions pour obtenir MovieLens 25M
|   |-- (les fichiers CSV ne sont pas versiones : trop volumineux)
|
|-- src/
|   |-- matrix_utils.py               # Construction et manipulation de la matrice sparse
|   |-- collaborative_filtering.py    # CF item-based et user-based
|   |-- svd_recommender.py            # Factorisation SVD et projection
|   |-- visualization.py             # Fonctions de visualisation UMAP et graphiques
|
|-- figures/
|   |-- matrix_heatmap_sparsity.png   # Heatmap + scatter de la matrice sparse
|   |-- ratings_distribution.png      # Distribution des notes
|   |-- user_movie_distributions.png  # Distributions activite utilisateurs/films
|   |-- singular_values.png           # Valeurs singulieres du SVD
|   |-- cumulative_variance.png       # Variance expliquee cumulee
|   |-- umap_all_movies.png           # Projection UMAP globale
|   |-- umap_anchor_highlighted.png   # Projection avec films d'ancrage
|   |-- umap_genre_distribution.png   # Distribution par genre dans l'espace latent
|
|-- requirements.txt                  # Dependances Python
|-- .gitignore
```

---

## 10. Reproductibilite

### Dependances

```
numpy>=1.24
pandas>=1.5
scipy>=1.10
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
umap-learn>=0.5
```

Installation :

```bash
pip install -r requirements.txt
```

### Donnees

Le jeu de donnees MovieLens 25M est disponible gratuitement sur le site du GroupLens Research Lab :

```
https://grouplens.org/datasets/movielens/25m/
```

Apres telechargement, placer les fichiers `movies.csv` et `ratings.csv` dans le repertoire `data/` et adapter les chemins de chargement dans le notebook.

### Reproductibilite numerique

La graine aleatoire est fixee a 42 pour l'ensemble du projet :

```python
np.random.seed(42)
# TruncatedSVD(random_state=42)
# UMAP(random_state=42)
```

Les resultats numeriques (valeurs singulieres, coordonnees UMAP, scores de similarite) sont donc reproductibles a l'identique sous reserve d'utiliser les memes versions de librairies.

---

*Projet realise dans le cadre du Master 2 Machine Learning — Universite de Strasbourg*  
*Dataset : MovieLens 25M (GroupLens, University of Minnesota)*
