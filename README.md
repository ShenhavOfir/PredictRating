# Recommender System

This repository contains Python code for various recommender systems to predict user ratings for items using different algorithms: baseline, neighborhood-based, least squares, and competition recommenders.

## Key Classes

### `Recommender`

Abstract base class for all recommenders.
- **Methods**:
  - `initialize_predictor(ratings: pd.DataFrame)`: Initialize the predictor.
  - `predict(user: int, item: int, timestamp: int) -> float`: Predict the rating.
  - `rmse(true_ratings: pd.DataFrame) -> float`: Calculate RMSE.

### `BaselineRecommender`

Uses baseline estimates for predictions.
- **Methods**:
  - `initialize_predictor(ratings: pd.DataFrame)`: Initialize user and item biases.
  - `predict(user: int, item: int, timestamp: int) -> float`: Predict using the baseline model.

### `NeighborhoodRecommender`

Uses user-user similarity for predictions.
- **Methods**:
  - `initialize_predictor(ratings: pd.DataFrame)`: Initialize user similarities.
  - `predict(user: int, item: int, timestamp: int) -> float`: Predict using user similarity.

### `LSRecommender`

Uses least squares regression for predictions.
- **Methods**:
  - `initialize_predictor(ratings: pd.DataFrame)`: Initialize least squares model.
  - `predict(user: int, item: int, timestamp: int) -> float`: Predict using least squares.

### `CompetitionRecommender`

Placeholder for advanced recommender.
- **Methods**:
  - `initialize_predictor(ratings: pd.DataFrame)`: Initialize the predictor.
  - `predict(user: int, item: int, timestamp: int) -> float`: Predict the rating.


