from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/CPP-BT_2026-01-17_DE_PQT_Reden_Gesamt.parquet")
OUT_PATH = Path("data/processed/02_cppbt_cleaned.csv")
START_DATE = "2017-10-24"
MIN_WORD_COUNT = 200

PARTY_MAP = {
    "cdu/csu": "cdu/csu",
    "spd": "spd",
    "bündnis 90/die grünen": "grüne",
    "die grünen": "grüne",
    "fdp": "fdp",
    "die linke": "linke",
    "afd": "afd",
}

KEEP_COLS = [
    "doc_id",
    "rede_id",
    "wahlperiode",
    "sitzung_nr",
    "sitzung_datum",
    "sitzung_jahr",
    "protokoll_nr",
    "protokoll_seite",
    "redner_id",
    "speaker_name",
    "redner_fraktion",
    "party_clean",
    "redner_rolle_kurz",
    "redner_rolle_lang",
    "rede_text",
    "zeichen",
    "tokens",
    "saetze",
    "word_count",
]


def clean_text(series: pd.Series) -> pd.Series:
    """Apply the same text normalization as in the original script."""
    return (
        series.astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.replace("\t", " ", regex=False)
        .str.replace("-", "-", regex=False)
        .str.replace("–", "-", regex=False)
        .str.replace("—", "-", regex=False)
        .str.replace("  ", " ", regex=False)
        .str.strip()
    )


def main() -> None:
    df = pd.read_parquet(RAW_PATH)

    df["sitzung_datum"] = pd.to_datetime(df["sitzung_datum"], errors="coerce")
    df = df[df["sitzung_datum"] >= START_DATE].copy()

    df["party"] = df["redner_fraktion"].astype(str).str.lower().str.strip()
    df = df[df["party"].notna()].copy()
    df = df[df["party"] != ""].copy()
    df = df[df["party"] != "none"].copy()

    df["party_clean"] = df["party"].map(PARTY_MAP)
    df = df[df["party_clean"].notna()].copy()

    df["word_count"] = df["rede_text"].astype(str).str.split().str.len()
    df = df[df["word_count"] >= MIN_WORD_COUNT].copy()

    df["speaker_name"] = (
        df["redner_vorname"].fillna("") + " " + df["redner_nachname"].fillna("")
    ).str.strip()

    df["rede_text"] = clean_text(df["rede_text"])
    df = df[KEEP_COLS].copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("=" * 80)
    print("CPP-BT bereinigt")
    print("=" * 80)
    print("Form:", df.shape)
    print("\nParteien:")
    print(df["party_clean"].value_counts())
    print("\nWahlperioden:")
    print(df["wahlperiode"].value_counts().sort_index())
    print("\nZeitraum:")
    print(df["sitzung_datum"].min(), "bis", df["sitzung_datum"].max())
    print("\nGespeichert unter:", OUT_PATH)


if __name__ == "__main__":
    main()
