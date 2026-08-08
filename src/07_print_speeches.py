from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/03_cppbt_government.csv")
OUT_PATH = Path("outputs/testreden.txt")

REDE_IDS = [
    "ID19100400",
    "ID194806300",
    "ID198702700",
    "ID1912406100",
    "ID1916104900",
    "ID1919507200",
    "ID1922808100",
    "ID203204700",
    "ID206705800",
    "ID2010704400",
    "ID2014405400",
    "ID2018116900",
    "ID211012700",
    "ID213206100",
    "ID215316100",
    "ID19100600",
    "ID194606900",
    "ID198317900",
    "ID1911805500",
    "ID1915202000",
    "ID1918403900",
    "ID1921603300",
    "ID20400800",
    "ID204713100",
    "ID208211900",
    "ID2011805800",
    "ID2015312000",
    "ID2018512500",
    "ID211814300",
    "ID215316500",
    "ID19100700",
    "ID194507500",
    "ID198009800",
    "ID1911500400",
    "ID1914511600",
    "ID1917612800",
    "ID1920904800",
    "ID1923316200",
    "ID203113300",
    "ID206103000",
    "ID209112900",
    "ID2012201200",
    "ID2015006700",
    "ID2017811500",
    "ID2021405600",
    "ID19100800",
    "ID195505500",
    "ID199900600",
    "ID1914009600",
    "ID1918103300",
    "ID1921908600",
    "ID202107300",
    "ID205002800",
    "ID207603500",
    "ID2010906400",
    "ID2013702800",
    "ID2016409800",
    "ID2019312600",
    "ID211902000",
    "ID215316300",
    "ID19100500",
    "ID194219000",
    "ID197700500",
    "ID1910803300",
    "ID1914006300",
    "ID1917212100",
    "ID1920209200",
    "ID1922800500",
    "ID202710100",
    "ID206307800",
    "ID209803900",
    "ID2013120000",
    "ID2018815200",
    "ID212115900",
    "ID215316400",
    "ID19100300",
    "ID195210100",
    "ID199409800",
    "ID1913308900",
    "ID1917312200",
    "ID1921106700",
    "ID20300100",
    "ID204107900",
    "ID207002200",
    "ID2010009800",
    "ID2012815600",
    "ID2015709100",
    "ID2018604100",
    "ID211409100",
    "ID215316000",
]


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as file:
        for rede_id in REDE_IDS:
            row = df[df["rede_id"] == rede_id]

            file.write("=" * 100 + "\n")
            file.write(f"REDE_ID: {rede_id}\n")
            file.write("=" * 100 + "\n\n")
            file.write(row["rede_text"].iloc[0])
            file.write("\n\n\n")

    print(f"Datei gespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
