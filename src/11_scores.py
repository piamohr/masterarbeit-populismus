from pathlib import Path
import pandas as pd


# =====================================================
# EINSTELLUNGEN
# =====================================================

INPUT_FILE = Path("data/processed/final_llm_coding/final_llm_scores_analysis_ready.csv")
OUTPUT_FILE = Path("data/processed/final_llm_coding/final_llm_scores_with_indices.csv")

POP_COLS = [
    "P1_people",
    "P2_anti_elite",
    "P3_outgroup",
]

GAL_TAN_COLS = [
    "G1_environment",
    "G2_migration",
    "G3_society",
    "G4_civil_rights",
    "G5_cosmopolitanism",
]

REQUIRED_COLS = POP_COLS + GAL_TAN_COLS


# =====================================================
# FUNKTIONEN
# =====================================================

def validate_columns(df: pd.DataFrame) -> None:
    """Prüft, ob alle für die Indexbildung benötigten Spalten vorhanden sind."""

    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Diese Spalten fehlen im Input: {missing_cols}")


def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet Populismus- und GAL-TAN-Indizes ohne Veränderung der Originalspalten."""

    df = df.copy()

    # Scores numerisch machen.
    for col in REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Populismus-Index
    # Mittelwert über die drei Populismusdimensionen.
    # Skala: 0 bis 2
    df["populism_score"] = df[POP_COLS].mean(axis=1, skipna=True)

    # Summenindex über die drei Populismusdimensionen.
    # Skala: 0 bis 6
    df["populism_sum"] = df[POP_COLS].sum(axis=1, skipna=True)

    # Anzahl gültiger Populismusdimensionen
    df["populism_n_valid"] = df[POP_COLS].notna().sum(axis=1)

    # GAL-TAN-Index
    # Mittelwert über alle substanziell codierten GAL-TAN-Dimensionen.
    # Skala: 0 bis 10
    # Wichtig: Fehlende Werte werden nicht ersetzt, sondern ignoriert.
    df["gal_tan_score"] = df[GAL_TAN_COLS].mean(axis=1, skipna=True)

    # Anzahl gültiger GAL-TAN-Dimensionen pro Rede
    # Skala: 0 bis 5
    df["gal_tan_n_valid"] = df[GAL_TAN_COLS].notna().sum(axis=1)

    # Markierung, ob überhaupt ein GAL-TAN-Wert vorliegt
    df["has_gal_tan_score"] = df["gal_tan_n_valid"] > 0

    return df


def print_checks(df: pd.DataFrame) -> None:
    """Gibt dieselben Kontrollausgaben wie das ursprüngliche Skript aus."""

    print("\nPopulismus-Score:")
    print(df["populism_score"].describe())

    print("\nGAL-TAN-Score:")
    print(df["gal_tan_score"].describe())

    print("\nAnzahl gültiger GAL-TAN-Dimensionen pro Rede:")
    print(df["gal_tan_n_valid"].value_counts().sort_index())

    print("\nAnteil Reden ohne GAL-TAN-Score:")
    print(df["gal_tan_score"].isna().mean())


# =====================================================
# MAIN
# =====================================================

def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    validate_columns(df)

    print("Anzahl Reden:", len(df))
    print("Anzahl Spalten:", len(df.columns))

    df = add_indices(df)

    print_checks(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nGespeichert unter: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
