# Movie Recommendation System

A Python project that recommends movies based on what similar users have watched and rated — built using collaborative filtering and K-Nearest Neighbors.

---

## What This Does

The idea is simple: if two movies tend to get rated similarly by the same group of users, they're probably alike in some way. This project uses that logic to suggest movies you might enjoy based on one you already like.

Given a movie title, the system finds the top `k` most similar movies by comparing rating patterns across all users — using cosine similarity under the hood.

---

## Dataset

The project uses a `rating.csv` file with the following columns:

| Column    | Description                       |
|-----------|-----------------------------------|
| `userId`  | Unique ID for each user           |
| `movieId` | Unique ID for each movie          |
| `title`   | Movie title                       |
| `rating`  | Rating given by the user (0.5–5)  |

The dataset comes from [MovieLens](https://grouplens.org/datasets/movielens/), a well-known benchmark for recommendation systems.

---

## Libraries Used

- `pandas` — loading and wrangling the data
- `numpy` — general numerical operations
- `scipy` — building the sparse matrix
- `scikit-learn` — running the KNN model
- `matplotlib` / `seaborn` — any visualizations
- Google Colab — where the notebook runs

---

## How It Works

1. Load the ratings data from CSV
2. Map each movie title and user ID to a numeric index
3. Build a sparse matrix (movies × users) using those indices and the rating values
4. Fit a `NearestNeighbors` model with cosine similarity on that matrix
5. For a given movie, pull its row from the matrix and find the closest neighbors

---

## Running It

Install the dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

Load the data:

```python
import pandas as pd
from scipy.sparse import csr_matrix

ratings = pd.read_csv("rating.csv")
ratings.columns = ratings.columns.str.strip()
```

Build the matrix:

```python
X, movie_mapper, movie_inv_mapper = create_matrix(ratings)
```

Get recommendations:

```python
recommendations = recommend_movies("Toy Story (1995)", X, movie_mapper, movie_inv_mapper, k=5)
print(recommendations)
```

Output:

```
['Toy Story 2 (1999)', "A Bug's Life (1998)", 'Monsters, Inc. (2001)', ...]
```

---

## Project Layout

```
movie-recommender/
├── rating.csv               # the dataset
├── recommendation.ipynb     # main notebook
└── README.md
```

---

## Key Functions

**`create_matrix(df)`**  
Takes the ratings DataFrame and converts it into a sparse CSR matrix. Returns the matrix along with two dictionaries for mapping movie titles to indices and back.

**`recommend_movies(movie_title, X, movie_mapper, movie_inv_mapper, k=5)`**  
Looks up the movie in the matrix, runs KNN, and returns a list of `k` similar movie titles. Returns an error message if the title isn't in the dataset
