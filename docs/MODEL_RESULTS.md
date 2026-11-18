# Model results (course content: Sessions 4, 6 and 8)

Produced by `python -m src.evaluate_models` on 75,607 Madrid flats (idealista18, 2018), 29 features.
All models predict log(price). Charts for the slides are in `docs/figures/`.

## 1. Model comparison: 5-fold cross-validation (Session 6)

| Model | Avg. error (MAPE) | ± across folds | MAE | R² (log price) | Training MAPE | Overfitting gap |
|---|---|---|---|---|---|---|
| **Random forest** (Session 8) | **13.0%** | 0.2% | ~47,100 € | 0.94 | 7.4% | 5.6 pts |
| Lasso (Session 6) | 25.1% | 0.2% | ~101,900 € | 0.77 | 25.1% | 0.0 pts |
| Linear regression (Session 4) | 25.1% | 0.2% | ~101,900 € | 0.77 | 25.1% | 0.0 pts |
| Ridge (Session 6) | 25.2% | 0.2% | ~101,500 € | 0.77 | 25.1% | 0.0 pts |

Chart: `figures/model_comparison.png`

**What it means**
- The random forest **halves the error** of every linear model, and the result is stable: all 5 folds are within ±0.2%.
- **Regularization does not help here.** Linear regression shows no overfitting (training error = test error), so there is nothing for Ridge/Lasso to fix. Lasso chose almost no penalty and kept all 29 features; Ridge's penalty barely changed the result. With 75,000 flats and only 29 features, the linear model's problem is **bias, not variance**: it is too simple to capture how location, size and features interact.
- The random forest captures those non-linear interactions (e.g. a terrace is worth more in Salamanca than in Vallecas), which is why it wins.

## 2. Overfitting check and tuning (Session 6: bias-variance trade-off)

| Min. flats per leaf | 1 | 2 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| Training MAPE | 4.6% | 6.0% | 7.4% | 9.3% | 11.6% | 13.6% | 15.6% |
| Test MAPE | 12.7% | 12.8% | 13.1% | 13.5% | 14.3% | 15.1% | 16.4% |

Chart: `figures/overfitting_check.png`

**What it means**
- Small leaves = complex trees: training error falls much faster than test error. The growing gap is **overfitting** (the trees memorise training flats).
- Even so, test error is still lowest with small leaves, because averaging many trees (bagging) controls the variance.
- **Our choice: leaf = 3** (13.1%). It is only 0.4 points worse than leaf = 1 but gives a much smaller model file and a smaller overfitting gap.

## 3. Price drivers (Session 8: feature importance)

Size (44%), bathrooms (14%), location north–south (12%), lift (9%), distance to Castellana (7%) and distance to the centre (4%) explain most decisions. Chart: `figures/feature_importance.png`

## 4. Predictions vs. reality

On a 20% test set the model never saw, predictions follow real prices closely across the whole range (50k€ to 2M€). Chart: `figures/actual_vs_predicted.png`

## Limitations to mention
- Prices are 2018 asking prices; today's level is applied with the ×1.75 market update.
- Asking price ≠ final sale price.
- Errors are larger for unusual flats (very large, luxury, very small).
