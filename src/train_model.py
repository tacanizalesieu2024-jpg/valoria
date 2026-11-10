"""Train the price model.   Run:  python -m src.train_model

Trains two models from the course and compares them on the same test set:
  1. Linear regression  (Session 4) - the baseline
  2. Random forest      (Session 8) - the main model
Both predict log(price): prices are very skewed, and the log makes errors
proportional (a 20k error matters more on a 150k flat than on a 1.5M one).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, MODEL_PATH
from src.features import FEATURES, clean_training_data


def evaluate(name, model, X_test, y_test):
    pred = np.exp(model.predict(X_test))
    mae = mean_absolute_error(y_test, pred)
    mape = mean_absolute_percentage_error(y_test, pred)
    print(f"{name:<20} MAE = {mae:>10,.0f} EUR   |   MAPE = {mape:.1%}")
    return mae


def main():
    df = clean_training_data(pd.read_csv(DATA_PATH))
    print(f"Training on {len(df):,} Madrid flats (idealista18, 2018)\n")

    X, y = df[FEATURES], df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    linear = LinearRegression().fit(X_train, np.log(y_train))
    forest = RandomForestRegressor(
        n_estimators=150, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=42
    ).fit(X_train, np.log(y_train))

    evaluate("Linear regression", linear, X_test, y_test)
    evaluate("Random forest", forest, X_test, y_test)

    importance = pd.Series(forest.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nTop 8 price drivers (random forest):")
    print(importance.head(8).round(3).to_string())

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": forest, "features": FEATURES, "train_data": df}, MODEL_PATH, compress=3)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
