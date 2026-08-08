from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/CPP-BT_2026-01-17_DE_PQT_Reden_Gesamt.parquet")
SAMPLE_PATH = Path("data/processed/01_cppbt_sample.csv")

REQUIRED_COLUMNS = [
    "doc_id",
    "rede_id",
    "wahlperiode",
    "sitzung_nr",
    "sitzung_datum",
    "sitzung_jahr",
    "protokoll_nr",
    "protokoll_seite",
    "redner_id",
    "redner_vorname",
    "redner_nachname",
    "redner_fraktion",
    "redner_rolle_kurz",
    "redner_rolle_lang",
    "rede_text",
    "zeichen",
    "tokens",
    "saetze",
]


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Check whether all explicitly expected columns are present."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Fehlende Spalten in der Rohdatei: {missing_columns}")


def main() -> None:
    df = pd.read_parquet(RAW_PATH)
    validate_columns(df, REQUIRED_COLUMNS)

    print("=" * 80)
    print("CPP-BT Reden geladen")
    print("=" * 80)
    print("Form:", df.shape)
    print("\nSpalten:")
    print(df.columns.tolist())
    print("\nErste Zeilen:")
    print(df.head())
    print("\nDatentypen:")
    print(df.dtypes)
    print("\nDatensatzgröße:")
    print(df.shape)

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.head(1000).to_csv(SAMPLE_PATH, index=False)
    print(f"\nStichprobe gespeichert: {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
