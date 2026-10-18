"""Train the FaceVitals models from dataset_full.csv and save them to
models/vitals_models.pkl. Run: python scripts/train_models.py"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["heart_rate_bpm", "hrv_sdnn_ms", "hrv_rmssd_ms", "upstroke_feature",
            "perfusion_index", "age_years", "bmi", "sex_M1_F0",
            "smoker", "family_history", "activity_level"]


def main():
    df = pd.read_csv(ROOT / "dataset_full.csv")
    X = df[FEATURES].to_numpy()
    y_bp = df[["systolic_bp_mmhg", "diastolic_bp_mmhg"]].to_numpy()
    y_met = df[["cholesterol_mg_dl", "glucose_mg_dl"]].to_numpy()

    X_train, X_test, bp_train, bp_test, met_train, met_test = train_test_split(
        X, y_bp, y_met, test_size=0.2, random_state=1)

    bp_model = RandomForestRegressor(n_estimators=50, max_depth=7,
                                     min_samples_leaf=4, n_jobs=-1,
                                     random_state=1).fit(X_train, bp_train)
    met_model = RandomForestRegressor(n_estimators=50, max_depth=7,
                                      min_samples_leaf=4, n_jobs=-1,
                                      random_state=1).fit(X_train, met_train)

    print("Blood Pressure (BP)  -> Systolic (SBP) Mean Absolute Error (MAE):",
          round(mean_absolute_error(bp_test[:, 0], bp_model.predict(X_test)[:, 0]), 2),
          "| Diastolic (DBP) MAE:",
          round(mean_absolute_error(bp_test[:, 1], bp_model.predict(X_test)[:, 1]), 2))
    print("Metabolic            -> Cholesterol MAE:",
          round(mean_absolute_error(met_test[:, 0], met_model.predict(X_test)[:, 0]), 2),
          "| Glucose MAE:",
          round(mean_absolute_error(met_test[:, 1], met_model.predict(X_test)[:, 1]), 2))

    out = ROOT / "models" / "vitals_models.pkl"
    with open(out, "wb") as f:
        pickle.dump({"bp": bp_model, "metabolic": met_model}, f)
    print("Saved:", out)


if __name__ == "__main__":
    main()
