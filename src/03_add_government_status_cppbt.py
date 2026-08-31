from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/02_cppbt_cleaned.csv")
OUT_PATH = Path("data/processed/03_cppbt_government.csv")

REQUIRED_COLUMNS = [
    "wahlperiode",
    "party_clean",
]

GOVERNMENT_PARTIES = {
    19: {"cdu/csu", "spd"},
    20: {"spd", "grüne", "fdp"},
    21: {"cdu/csu", "spd"},
}

GOVERNMENT_LABELS = {
    1: "Regierung",
    0: "Opposition",
}


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Check whether all explicitly expected columns are present."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Fehlende Spalten in {DATA_PATH}: {missing_columns}")


def get_government_status(row: pd.Series) -> int | None:
    wahlperiode = row["wahlperiode"]
    party = row["party_clean"]

    if wahlperiode not in GOVERNMENT_PARTIES:
        return None

    return 1 if party in GOVERNMENT_PARTIES[wahlperiode] else 0


def main() -> None:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    validate_columns(df, REQUIRED_COLUMNS)

    df["government_status"] = df.apply(get_government_status, axis=1)
    df["government_label"] = df["government_status"].map(GOVERNMENT_LABELS)

    print(pd.crosstab(df["wahlperiode"], df["government_label"]))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nGespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
