from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/03_cppbt_government.csv")
OUT_PATH = Path("data/processed/06_test_sample.csv")
N_SAMPLE = 10
RANDOM_STATE = 47


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    test_df = df.sample(n=N_SAMPLE, random_state=RANDOM_STATE)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    main()
