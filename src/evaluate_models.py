"""Compare models the way the course teaches (Sessions 4, 6 and 8).   Run:  python -m src.evaluate_models

1. 5-fold cross-validation (Session 6) for four models:
      Linear regression (Session 4)  ·  Ridge and Lasso (Session 6, regularization)  ·  Random forest (Session 8)
2. Overfitting check: training error vs. test error for each model.
3. Tuning the random forest: test error for different minimum leaf sizes (bias-variance trade-off).
4. Charts for the slides, saved in docs/figures/, and a results table in docs/model_results.csv.

All models predict log(price); errors are reported in euros and % after converting back.
Takes about 3-6 minutes on a laptop.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PATH, ROOT
from src.features import FEATURES, clean_training_data
from src.valuation import READABLE

OUT = ROOT / "docs"
FIG = OUT / "figures"
TEAL, TERRACOTTA, GREY, INK, MUTED = "#00957A", "#C9573A", "#B8BEC3", "#1B2733", "#6B7280"


def models():
    """The four models. Ridge/Lasso need scaled features; the alpha (penalty) is chosen by internal CV."""
    return {
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "Lasso": make_pipeline(StandardScaler(), LassoCV(alphas=np.logspace(-5, 0, 30), cv=3, max_iter=5000, random_state=42)),
        "Random forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features=0.5,
                                               n_jobs=-1, random_state=42),
    }


def scores(y_true, log_pred):
    pred = np.exp(log_pred)
    return {"MAE (€)": mean_absolute_error(y_true, pred),
            "MAPE": mean_absolute_percentage_error(y_true, pred),
            "R² (log price)": r2_score(np.log(y_true), log_pred)}


def cross_validate(X, y, k=5):
    """5-fold CV: each flat is in the test fold exactly once. Also records training error (overfitting)."""
    rows = []
    folds = KFold(n_splits=k, shuffle=True, random_state=42)
    for name, _ in models().items():
        for i, (tr, te) in enumerate(folds.split(X)):
            m = models()[name].fit(X.iloc[tr], np.log(y.iloc[tr]))
            test = scores(y.iloc[te], m.predict(X.iloc[te]))
            train = scores(y.iloc[tr], m.predict(X.iloc[tr]))
            rows.append({"model": name, "fold": i + 1, **test,
                         "train MAPE": train["MAPE"], "train MAE (€)": train["MAE (€)"]})
            print(f"  {name:<18} fold {i + 1}: test MAPE {test['MAPE']:.1%} | train MAPE {train['MAPE']:.1%}")
    return pd.DataFrame(rows)


def tune_forest(X_tr, y_tr, X_te, y_te):
    """Bias-variance trade-off: small leaves = complex trees (low train error, risk of overfitting)."""
    rows = []
    for leaf in [1, 2, 3, 5, 10, 20, 50]:
        m = RandomForestRegressor(n_estimators=100, min_samples_leaf=leaf, max_features=0.5,
                                  n_jobs=-1, random_state=42).fit(X_tr, np.log(y_tr))
        rows.append({"min_samples_leaf": leaf,
                     "train MAPE": scores(y_tr, m.predict(X_tr))["MAPE"],
                     "test MAPE": scores(y_te, m.predict(X_te))["MAPE"]})
        print(f"  min_samples_leaf={leaf:<3} train {rows[-1]['train MAPE']:.1%} | test {rows[-1]['test MAPE']:.1%}")
    return pd.DataFrame(rows)


# ---------------- Charts (thin marks, recessive axes, labels in text colours) ----------------
def _style(ax, title, subtitle=None):
    import matplotlib.pyplot as plt  # noqa: F401
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#D9DDE1")
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=INK, pad=24 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=MUTED)


def chart_model_comparison(summary):
    import matplotlib.pyplot as plt
    s = summary.sort_values("MAPE")
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200)
    colours = [TEAL if n == "Random forest" else GREY for n in s.index]
    ax.barh(s.index[::-1], s["MAPE"][::-1] * 100, color=colours[::-1], height=0.55)
    for i, (name, row) in enumerate(s[::-1].iterrows()):
        ax.text(row["MAPE"] * 100 + 0.4, i, f"{row['MAPE']:.1%}  (± {row['MAPE std']:.1%})",
                va="center", fontsize=10, color=INK)
    ax.set_xlabel("Average error on unseen flats (MAPE, %) — lower is better", color=MUTED, fontsize=10)
    ax.grid(axis="x", color="#EEF0F2", linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax, "Random forest halves the error of linear models",
           "5-fold cross-validation on 75,600 Madrid flats (± = spread across folds)")
    ax.set_xlim(0, s["MAPE"].max() * 100 * 1.35)
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison.png", facecolor="white")
    plt.close(fig)


def chart_actual_vs_predicted(y_true, y_pred):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    ax.scatter(y_true / 1e3, y_pred / 1e3, s=4, alpha=0.18, color=TEAL, linewidths=0)
    lim = [30, 3000]
    ax.plot(lim, lim, color=INK, linewidth=1, linestyle="--")
    ax.text(1900, 2300, "perfect prediction", color=MUTED, fontsize=9, rotation=45, ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ticks = [50, 100, 200, 500, 1000, 2000]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Actual asking price (k€, 2018, log scale)", color=MUTED, fontsize=10)
    ax.set_ylabel("Predicted price (k€, log scale)", color=MUTED, fontsize=10)
    _style(ax, "Predictions follow real prices closely",
           "Random forest on a 20% test set the model never saw")
    fig.tight_layout()
    fig.savefig(FIG / "actual_vs_predicted.png", facecolor="white")
    plt.close(fig)


def chart_feature_importance(model):
    import matplotlib.pyplot as plt
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False).head(10)
    names = [READABLE.get(f, f.replace("_", " ").capitalize()) for f in imp.index]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    ax.barh(names[::-1], imp.values[::-1] * 100, color=TEAL, height=0.55)
    for i, v in enumerate(imp.values[::-1]):
        ax.text(v * 100 + 0.5, i, f"{v:.0%}", va="center", fontsize=10, color=INK)
    ax.set_xlabel("Share of the model's decisions (%)", color=MUTED, fontsize=10)
    ax.grid(axis="x", color="#EEF0F2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, imp.max() * 100 * 1.15)
    _style(ax, "Size and location drive most of the price",
           "Random forest feature importance (Session 8), top 10 of 29 features")
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png", facecolor="white")
    plt.close(fig)


def chart_overfitting(tuning):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    x = tuning["min_samples_leaf"]
    ax.plot(x, tuning["train MAPE"] * 100, color=TEAL, linewidth=2, marker="o", markersize=6)
    ax.plot(x, tuning["test MAPE"] * 100, color=TERRACOTTA, linewidth=2, marker="s", markersize=6,
            linestyle="--")
    ax.text(x.iloc[-1] * 1.08, tuning["train MAPE"].iloc[-1] * 100, "Training error", color=INK,
            fontsize=10, va="center")
    ax.text(x.iloc[-1] * 1.08, tuning["test MAPE"].iloc[-1] * 100, "Test error", color=INK,
            fontsize=10, va="center")
    best = tuning.loc[tuning["test MAPE"].idxmin()]
    ax.annotate(f"lowest test error: {best['test MAPE']:.1%} (leaf = {int(best['min_samples_leaf'])})",
                (best["min_samples_leaf"], best["test MAPE"] * 100), xytext=(6, 12),
                textcoords="offset points", ha="left", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlim(0.8, x.iloc[-1] * 2.2)
    ax.set_xlabel("Minimum flats per leaf (higher = simpler trees)", color=MUTED, fontsize=10)
    ax.set_ylabel("Average error (MAPE, %)", color=MUTED, fontsize=10)
    ax.grid(axis="y", color="#EEF0F2", linewidth=0.8)
    ax.legend(["Training error", "Test error"], frameon=False, fontsize=9, loc="lower right",
              labelcolor=INK)
    _style(ax, "Training error falls faster than test error: that gap is overfitting",
           "Random forest tuning on an 80/20 split; we use leaf = 3 (13.1%) to keep the model small")
    fig.tight_layout()
    fig.savefig(FIG / "overfitting_check.png", facecolor="white")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = clean_training_data(pd.read_csv(DATA_PATH))
    X, y = df[FEATURES], df["price"]
    print(f"Data: {len(df):,} flats, {len(FEATURES)} features\n\n1) 5-fold cross-validation")

    cv = cross_validate(X, y)
    summary = cv.groupby("model").agg(**{
        "MAE (€)": ("MAE (€)", "mean"), "MAPE": ("MAPE", "mean"), "MAPE std": ("MAPE", "std"),
        "R² (log price)": ("R² (log price)", "mean"), "train MAPE": ("train MAPE", "mean")})
    summary["overfitting gap"] = summary["MAPE"] - summary["train MAPE"]
    summary = summary.sort_values("MAPE")
    print("\nCross-validation summary (mean of 5 folds):")
    print(summary.round({"MAE (€)": 0, "MAPE": 3, "MAPE std": 3, "R² (log price)": 3,
                         "train MAPE": 3, "overfitting gap": 3}).to_string())

    # Ridge/Lasso: which penalty was chosen, and how many features Lasso kept
    fitted = {n: m.fit(X, np.log(y)) for n, m in models().items() if n in ("Ridge", "Lasso")}
    ridge_alpha = fitted["Ridge"][-1].alpha_
    lasso = fitted["Lasso"][-1]
    kept = int((np.abs(lasso.coef_) > 1e-6).sum())
    print(f"\nRidge chose alpha = {ridge_alpha:.3g}; Lasso chose alpha = {lasso.alpha_:.3g} "
          f"and kept {kept} of {len(FEATURES)} features")

    print("\n2) Random forest tuning (80/20 split)")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    tuning = tune_forest(X_tr, y_tr, X_te, y_te)

    forest = models()["Random forest"].fit(X_tr, np.log(y_tr))
    pred = np.exp(forest.predict(X_te))

    summary.to_csv(OUT / "model_results.csv")
    tuning.to_csv(OUT / "forest_tuning.csv", index=False)
    chart_model_comparison(summary)
    chart_actual_vs_predicted(y_te.values, pred)
    chart_feature_importance(forest)
    chart_overfitting(tuning)
    print(f"\nSaved: docs/model_results.csv, docs/forest_tuning.csv and 4 charts in docs/figures/")


if __name__ == "__main__":
    main()
