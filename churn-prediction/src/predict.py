"""
Score de nouveaux clients avec le modèle de churn entraîné dans le notebook.

Usage :
    python src/predict.py --input data/telco_customer_churn.csv --output predictions.csv

Le fichier d'entrée doit avoir le même format que le jeu de données Telco d'origine
(la colonne `Churn` est facultative).
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

SERVICES = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Applique exactement le même nettoyage et feature engineering que le notebook."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    if pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["NumServices"] = (df[SERVICES] == "Yes").sum(axis=1) + (df["InternetService"] != "No").astype(int)
    return df.drop(columns=["customerID", "Churn"], errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prédiction du churn client")
    parser.add_argument("--input", required=True, help="CSV des clients à scorer")
    parser.add_argument("--output", default="predictions.csv", help="CSV de sortie")
    args = parser.parse_args()

    artifact = joblib.load(MODEL_PATH)
    pipeline, threshold = artifact["pipeline"], artifact["threshold"]

    raw = pd.read_csv(args.input)
    proba = pipeline.predict_proba(prepare(raw))[:, 1]

    result = pd.DataFrame({
        "customerID": raw["customerID"],
        "churn_probability": proba.round(4),
        "target_for_retention": proba >= threshold,
    }).sort_values("churn_probability", ascending=False)
    result.to_csv(args.output, index=False)

    print(f"Modèle : {artifact['model_name']}  |  seuil de décision : {threshold:.2f}")
    print(f"{len(result)} clients scorés, {result['target_for_retention'].sum()} à cibler.")
    print(f"Résultats enregistrés dans {args.output}\n")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
