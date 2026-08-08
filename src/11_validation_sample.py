from pathlib import Path
import numpy as np
import pandas as pd


INPUT_FILE = Path("data/processed/03_cppbt_government.csv")
OUTPUT_FILE = Path("data/processed/validation/validation_sample.csv")

PARTY_COLUMN = "party_clean"
DATE_COLUMN = "sitzung_datum"
N_PER_PARTY = 15

REQUIRED_COLUMNS = [
    PARTY_COLUMN,
    DATE_COLUMN,
]


def check_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Bricht mit klarer Fehlermeldung ab, falls erwartete Spalten fehlen."""
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "Folgende erwartete Spalten fehlen im Input-Datensatz: "
            + ", ".join(missing_columns)
        )


def draw_balanced_sample(group: pd.DataFrame, n: int = N_PER_PARTY) -> pd.DataFrame:
    """
    Zieht bis zu n Reden je Partei, gleichmäßig über den jeweiligen Zeitraum verteilt.

    Die Auswahl erfolgt deterministisch über äquidistante Positionen im nach Datum
    sortierten Parteidatensatz. Dadurch bleibt das Verhalten des Originalskripts erhalten.
    """
    group = group.sort_values(DATE_COLUMN)

    positions = np.linspace(
        0,
        len(group) - 1,
        min(n, len(group)),
        dtype=int,
    )

    return group.iloc[positions]


def main() -> None:
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    check_required_columns(df, REQUIRED_COLUMNS)

    validation_sample = (
        df.groupby(PARTY_COLUMN, group_keys=True)
        .apply(draw_balanced_sample, n=N_PER_PARTY)
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    validation_sample.to_csv(OUTPUT_FILE, index=False)

    print("\nSamplegröße:")
    print(len(validation_sample))

    print("\nVerteilung nach Partei:")
    print(validation_sample.groupby(PARTY_COLUMN).size())

    print("\nZeitspanne pro Partei:")
    print(validation_sample.groupby(PARTY_COLUMN)[DATE_COLUMN].agg(["min", "max"]))

    print(f"\nGespeichert: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
