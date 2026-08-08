from pathlib import Path
import pandas as pd


# =====================================================
# EINSTELLUNGEN
# =====================================================

METADATA_FILE = Path("data/processed/03_cppbt_government.csv")
LLM_FILE = Path("data/processed/final_llm_coding/final_llm_scores_with_indices.csv")

OUTPUT_FILE = Path("data/processed/final_llm_coding/final_analysis_dataset_merged.csv")

ID_COL = "rede_id"

META_COLS = [
    "rede_id",
    "doc_id",
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
    "government_status",
    "government_label",
]

LLM_COLS = [
    "rede_id",

    # Populismus-Dimensionen
    "P1_people",
    "P2_anti_elite",
    "P3_outgroup",

    # GAL-TAN-Dimensionen
    "G1_environment",
    "G2_migration",
    "G3_society",
    "G4_civil_rights",
    "G5_cosmopolitanism",

    # Indizes
    "populism_score",
    "populism_sum",
    "populism_n_valid",
    "gal_tan_score",
    "gal_tan_n_valid",
    "has_gal_tan_score",
]


# =====================================================
# FUNKTIONEN
# =====================================================

def validate_columns(df: pd.DataFrame, required_cols: list[str], label: str) -> None:
    """Prüft, ob alle erwarteten Spalten vorhanden sind."""

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Diese {label}-Spalten fehlen: {missing_cols}")


def prepare_dataframes(meta: pd.DataFrame, llm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wählt die benötigten Spalten aus und vereinheitlicht die Merge-ID."""

    validate_columns(meta, META_COLS, "Metadaten")
    validate_columns(llm, LLM_COLS, "LLM")

    meta = meta[META_COLS].copy()
    llm = llm[LLM_COLS].copy()

    meta[ID_COL] = meta[ID_COL].astype(str)
    llm[ID_COL] = llm[ID_COL].astype(str)

    meta["sitzung_datum"] = pd.to_datetime(meta["sitzung_datum"], errors="coerce")

    return meta, llm


def print_pre_merge_checks(meta: pd.DataFrame, llm: pd.DataFrame) -> None:
    """Gibt dieselben Checks vor dem Merge wie das ursprüngliche Skript aus."""

    print("\nDoppelte rede_id in Metadaten:", meta[ID_COL].duplicated().sum())
    print("Doppelte rede_id in LLM-Daten:", llm[ID_COL].duplicated().sum())

    meta_ids = set(meta[ID_COL])
    llm_ids = set(llm[ID_COL])

    print("\nReden in Metadaten:", len(meta_ids))
    print("Reden in LLM-Daten:", len(llm_ids))
    print("LLM-Reden ohne Metadaten:", len(llm_ids - meta_ids))
    print("Metadaten-Reden ohne LLM-Codierung:", len(meta_ids - llm_ids))


def print_post_merge_checks(df: pd.DataFrame) -> None:
    """Gibt dieselben Plausibilitätschecks wie das ursprüngliche Skript aus."""

    print("\nZeitraum:")
    print(df["sitzung_datum"].min(), "bis", df["sitzung_datum"].max())

    print("\nReden pro Partei:")
    print(df["party_clean"].value_counts(dropna=False))

    print("\nRegierung/Opposition:")
    print(df["government_label"].value_counts(dropna=False))

    print("\nPopulismus-Score:")
    print(df["populism_score"].describe())

    print("\nGAL-TAN-Score:")
    print(df["gal_tan_score"].describe())

    print("\nAnteil ohne GAL-TAN-Score:")
    print(df["gal_tan_score"].isna().mean())


# =====================================================
# MAIN
# =====================================================

def main() -> None:
    meta = pd.read_csv(METADATA_FILE)
    llm = pd.read_csv(LLM_FILE)

    print("Metadaten:", meta.shape)
    print("LLM-Daten:", llm.shape)

    meta, llm = prepare_dataframes(meta, llm)

    print_pre_merge_checks(meta, llm)

    df = meta.merge(
        llm,
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )

    print("\nGemergter Datensatz:", df.shape)

    print_post_merge_checks(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nGespeichert unter: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
