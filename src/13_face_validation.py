from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


MANUAL_FILE = Path("data/raw/manual_validation.csv")
LLM_FILE = Path("data/processed/validation/validation_llm_scores.csv")
OUTPUT_FILE = Path("data/processed/validation/validation_results.csv")

ID_COLUMN = "rede_id"

POP_ITEMS = [
    "P1_people",
    "P2_anti_elite",
    "P3_outgroup",
]

GAL_ITEMS = [
    "G1_environment",
    "G2_migration",
    "G3_society",
    "G4_civil_rights",
    "G5_cosmopolitanism",
]

REQUIRED_MANUAL_COLUMNS = [ID_COLUMN, *POP_ITEMS, *GAL_ITEMS]
REQUIRED_LLM_COLUMNS = [ID_COLUMN, *POP_ITEMS, *GAL_ITEMS]


def check_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Folgende erwartete Spalten fehlen in {dataset_name}: "
            + ", ".join(missing_columns)
        )


def load_data() -> pd.DataFrame:
    manual = pd.read_csv(MANUAL_FILE, sep=";")
    llm = pd.read_csv(LLM_FILE)

    check_required_columns(manual, REQUIRED_MANUAL_COLUMNS, str(MANUAL_FILE))
    check_required_columns(llm, REQUIRED_LLM_COLUMNS, str(LLM_FILE))

    return manual.merge(
        llm,
        on=ID_COLUMN,
        suffixes=("_manual", "_llm"),
    )


def validate_populism(df: pd.DataFrame) -> list[dict]:
    rows = []

    for item in POP_ITEMS:
        kappa = cohen_kappa_score(
            df[f"{item}_manual"],
            df[f"{item}_llm"],
            weights="quadratic",
        )

        rows.append(
            {
                "item": item,
                "type": "populism",
                "metric": "weighted_kappa",
                "value": round(kappa, 3),
                "n": len(df),
            }
        )

    return rows


def validate_gal_tan(df: pd.DataFrame) -> list[dict]:
    rows = []

    for item in GAL_ITEMS:
        manual_col = f"{item}_manual"
        llm_col = f"{item}_llm"

        manual_present = df[manual_col].notna()
        llm_present = df[llm_col].notna()

        presence_agreement = (manual_present == llm_present).mean()

        rows.append(
            {
                "item": item,
                "type": "gal-tan",
                "metric": "presence_agreement",
                "value": round(presence_agreement, 3),
                "n": len(df),
            }
        )

        both_coded = df[df[manual_col].notna() & df[llm_col].notna()].copy()

        if (
            len(both_coded) >= 2
            and both_coded[manual_col].nunique() > 1
            and both_coded[llm_col].nunique() > 1
        ):
            spearman_value = spearmanr(
                both_coded[manual_col],
                both_coded[llm_col],
            ).correlation
        else:
            spearman_value = np.nan

        rows.append(
            {
                "item": item,
                "type": "gal-tan",
                "metric": "spearman_positions_only",
                "value": round(spearman_value, 3) if pd.notna(spearman_value) else np.nan,
                "n": len(both_coded),
            }
        )

    return rows


def main() -> None:
    df = load_data()

    results = []
    results.extend(validate_populism(df))
    results.extend(validate_gal_tan(df))

    results_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\nValidation Results:")
    print(results_df)
    print(f"\nGespeichert: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
