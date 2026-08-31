from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# EINSTELLUNGEN
# =====================================================

SPEECH_FILE = Path("data/processed/final_llm_coding/final_analysis_dataset_merged.csv")
POLL_FILE = Path("data/raw/infratest_dimap.xlsx")

OUTPUT_DIR = Path("data/processed/final_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SPEECH_POLL = OUTPUT_DIR / "02_speech_level_with_polling.csv"
OUTPUT_PARTY_INTERVAL = OUTPUT_DIR / "03_party_polling_interval.csv"
OUTPUT_MAINSTREAM_INTERVAL = OUTPUT_DIR / "04_mainstream_polling_interval.csv"
OUTPUT_PARTY_DISTANCE = OUTPUT_DIR / "05_party_polling_interval_distances.csv"
OUTPUT_MAINSTREAM_DISTANCE = OUTPUT_DIR / "06_mainstream_polling_interval_distances.csv"

MAINSTREAM_PARTIES = ["linke", "grüne", "spd", "cdu/csu", "fdp"]
ALL_PARTIES = ["linke", "grüne", "spd", "cdu/csu", "fdp", "afd"]

POP_COLS = [
    "P1_people",
    "P2_anti_elite",
    "P3_outgroup"
]

GAL_TAN_COLS = [
    "G1_environment",
    "G2_migration",
    "G3_society",
    "G4_civil_rights",
    "G5_cosmopolitanism"
]

GAL_TAN_TOPIC_LABELS = {
    "G1_environment": "environment",
    "G2_migration": "migration",
    "G3_society": "society",
    "G4_civil_rights": "civil_rights",
    "G5_cosmopolitanism": "cosmopolitanism"
}

TOPIC_COUNT_COLS = [
    "n_speeches_environment",
    "n_speeches_migration",
    "n_speeches_society",
    "n_speeches_civil_rights",
    "n_speeches_cosmopolitanism"
]

TOPIC_MAP = {
    "environment": "n_speeches_environment",
    "migration": "n_speeches_migration",
    "society": "n_speeches_society",
    "civil_rights": "n_speeches_civil_rights",
    "cosmopolitanism": "n_speeches_cosmopolitanism"
}


def check_required_columns(data: pd.DataFrame, required_columns: list[str], file_label: str) -> None:
    """Prüft, ob alle erwarteten Variablen vorhanden sind."""
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"In {file_label} fehlen erwartete Variablen: "
            f"{missing_columns}"
        )

# =====================================================
# DATEN LADEN
# =====================================================

speeches = pd.read_csv(
    SPEECH_FILE,
    dtype={
        "rede_id": str,
        "redner_id": str,
        "redner_rolle_kurz": str
    },
    low_memory=False
)

polls = pd.read_excel(POLL_FILE)

print("Reden:", speeches.shape)
print("Umfragen:", polls.shape)

required_speech_columns = [
    "rede_id",
    "redner_id",
    "redner_rolle_kurz",
    "party_clean",
    "sitzung_datum",
    "wahlperiode",
    "government_label",
    "word_count",
    "has_gal_tan_score",
    "populism_score",
    "populism_sum",
    "gal_tan_score",
] + POP_COLS + GAL_TAN_COLS

check_required_columns(
    data=speeches,
    required_columns=required_speech_columns,
    file_label=str(SPEECH_FILE),
)

# =====================================================
# BEREINIGUNG
# =====================================================

# Spaltennamen vereinheitlichen
polls.columns = [c.strip().lower() for c in polls.columns]

# Erwartete Umfragespalten nach Vereinheitlichung prüfen
required_poll_columns = ["datum"] + ALL_PARTIES
check_required_columns(
    data=polls,
    required_columns=required_poll_columns,
    file_label=str(POLL_FILE),
)

# Parteiennamen vereinheitlichen
speeches["party_clean"] = (
    speeches["party_clean"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Datumsvariablen
speeches["sitzung_datum"] = pd.to_datetime(
    speeches["sitzung_datum"],
    errors="coerce"
)

# Excel-Datum robust einlesen
if pd.api.types.is_numeric_dtype(polls["datum"]):
    polls["poll_date"] = pd.to_datetime(
        polls["datum"],
        origin="1899-12-30",
        unit="D"
    )
else:
    polls["poll_date"] = pd.to_datetime(
        polls["datum"],
        errors="coerce"
    )

# Umfragewerte numerisch machen
for col in ALL_PARTIES:
    polls[col] = pd.to_numeric(polls[col], errors="coerce")

# LLM-Scores numerisch machen
for col in POP_COLS + GAL_TAN_COLS + ["populism_score", "populism_sum", "gal_tan_score"]:
    speeches[col] = pd.to_numeric(speeches[col], errors="coerce")

# Sortieren
polls = polls.sort_values("poll_date").reset_index(drop=True)
speeches = speeches.sort_values("sitzung_datum").reset_index(drop=True)

# =====================================================
# POLLING-INTERVALLE ERZEUGEN
# =====================================================

polls["next_poll_date"] = polls["poll_date"].shift(-1)
polls["poll_interval_id"] = np.arange(len(polls))

# AfD-Unterstützung am Beginn des Intervalls
polls["afd_support_t"] = polls["afd"]

# Nur vollständige Intervalle behalten
poll_intervals = polls.dropna(subset=["next_poll_date"]).copy()

print("\nPolling-Zeitraum:")
print(
    poll_intervals["poll_date"].min(),
    "bis",
    poll_intervals["next_poll_date"].max()
)

# =====================================================
# REDEN POLLING-INTERVALLEN ZUORDNEN
# =====================================================

speeches_with_polls = pd.merge_asof(
    speeches,
    poll_intervals[
        [
            "poll_interval_id",
            "poll_date",
            "next_poll_date",
            "afd_support_t",
            "cdu/csu",
            "afd",
            "spd",
            "grüne",
            "linke",
            "fdp"
        ]
    ].sort_values("poll_date"),
    left_on="sitzung_datum",
    right_on="poll_date",
    direction="backward",
    allow_exact_matches=True
)

# Sicherheit:
# Rede muss tatsächlich im Intervall [poll_date, next_poll_date) liegen.
speeches_with_polls = speeches_with_polls[
    (speeches_with_polls["sitzung_datum"] >= speeches_with_polls["poll_date"]) &
    (speeches_with_polls["sitzung_datum"] < speeches_with_polls["next_poll_date"])
].copy()

print("\nReden nach Polling-Matching:", speeches_with_polls.shape)
print("Nicht zugeordnete Reden:", len(speeches) - len(speeches_with_polls))

# =====================================================
# UMFRAGEDATEN INS LONG-FORMAT BRINGEN
# =====================================================

polls_long = poll_intervals.melt(
    id_vars=[
        "poll_interval_id",
        "poll_date",
        "next_poll_date",
        "afd_support_t"
    ],
    value_vars=ALL_PARTIES,
    var_name="party_clean",
    value_name="party_support_t"
)

polls_long = polls_long.sort_values(
    ["party_clean", "poll_date"]
).reset_index(drop=True)

# =====================================================
# MODERATOR: SUPPORT CHANGE
# =====================================================

# Veränderung gegenüber der vorherigen Infratest-dimap-Erhebung.
# Negative Werte = Verlust
# Positive Werte = Zugewinn

polls_long["party_support_t_minus_1"] = (
    polls_long
    .groupby("party_clean")["party_support_t"]
    .shift(1)
)

polls_long["party_support_change"] = (
    polls_long["party_support_t"] -
    polls_long["party_support_t_minus_1"]
)

polls_long["party_support_change_abs"] = (
    polls_long["party_support_change"].abs()
)

# =====================================================
# PARTEISPEZIFISCHE SUPPORT-VARIABLEN AN REDE-LEVEL HÄNGEN
# =====================================================

speeches_with_polls = speeches_with_polls.merge(
    polls_long[
        [
            "poll_interval_id",
            "party_clean",
            "party_support_t",
            "party_support_t_minus_1",
            "party_support_change",
            "party_support_change_abs"
        ]
    ],
    on=["poll_interval_id", "party_clean"],
    how="left",
    validate="many_to_one"
)

# =====================================================
# GAL-TAN-THEMENPRÄSENZ AUF REDEEBENE
# =====================================================

# Eine GAL-TAN-Dimension gilt als angesprochen,
# wenn ein gültiger Score vorliegt.
# NaN bedeutet: Thema in der Rede nicht substanziell vorhanden / nicht codierbar.

for col, label in GAL_TAN_TOPIC_LABELS.items():
    speeches_with_polls[f"has_{label}"] = (
        speeches_with_polls[col]
        .notna()
        .astype(int)
    )

# =====================================================
# SPEECH-LEVEL MIT POLLING SPEICHERN
# =====================================================

speeches_with_polls.to_csv(OUTPUT_SPEECH_POLL, index=False)

print(f"\nGespeichert: {OUTPUT_SPEECH_POLL}")

# =====================================================
# VOLLSTÄNDIGES PARTEI × POLLING-INTERVALL-GRID
# =====================================================

complete_party_grid = (
    poll_intervals[
        [
            "poll_interval_id",
            "poll_date",
            "next_poll_date",
            "afd_support_t"
        ]
    ]
    .assign(key=1)
    .merge(
        pd.DataFrame({"party_clean": ALL_PARTIES, "key": 1}),
        on="key"
    )
    .drop(columns="key")
)

# Support-Variablen an vollständiges Grid hängen
support_vars = polls_long[
    [
        "poll_interval_id",
        "party_clean",
        "party_support_t",
        "party_support_t_minus_1",
        "party_support_change",
        "party_support_change_abs"
    ]
].copy()

complete_party_grid = complete_party_grid.merge(
    support_vars,
    on=["poll_interval_id", "party_clean"],
    how="left",
    validate="one_to_one"
)

# =====================================================
# REDEWERTE AGGREGIEREN: PARTEI × INTERVALL
# =====================================================

speech_agg = (
    speeches_with_polls
    .groupby(
        [
            "poll_interval_id",
            "party_clean"
        ],
        as_index=False
    )
    .agg({
        # Indizes
        "populism_score": "mean",
        "populism_sum": "mean",
        "gal_tan_score": "mean",

        # Populismus-Subdimensionen
        "P1_people": "mean",
        "P2_anti_elite": "mean",
        "P3_outgroup": "mean",

        # GAL-TAN-Subdimensionen
        "G1_environment": "mean",
        "G2_migration": "mean",
        "G3_society": "mean",
        "G4_civil_rights": "mean",
        "G5_cosmopolitanism": "mean",

        # GAL-TAN-Themenhäufigkeiten
        "has_environment": "sum",
        "has_migration": "sum",
        "has_society": "sum",
        "has_civil_rights": "sum",
        "has_cosmopolitanism": "sum",

        # Fallzahlen / Checks
        "rede_id": "count",
        "has_gal_tan_score": "sum",
        "word_count": "mean"
    })
)

speech_agg = speech_agg.rename(columns={
    "populism_score": "mean_populism_score",
    "populism_sum": "mean_populism_sum",
    "gal_tan_score": "mean_gal_tan_score",

    "P1_people": "mean_P1_people",
    "P2_anti_elite": "mean_P2_anti_elite",
    "P3_outgroup": "mean_P3_outgroup",

    "G1_environment": "mean_G1_environment",
    "G2_migration": "mean_G2_migration",
    "G3_society": "mean_G3_society",
    "G4_civil_rights": "mean_G4_civil_rights",
    "G5_cosmopolitanism": "mean_G5_cosmopolitanism",

    "has_environment": "n_speeches_environment",
    "has_migration": "n_speeches_migration",
    "has_society": "n_speeches_society",
    "has_civil_rights": "n_speeches_civil_rights",
    "has_cosmopolitanism": "n_speeches_cosmopolitanism",

    "rede_id": "n_speeches",
    "has_gal_tan_score": "n_speeches_with_gal_tan",
    "word_count": "mean_word_count"
})

# Aggregierte Redewerte an vollständiges Partei-Intervall-Grid hängen
party_interval = complete_party_grid.merge(
    speech_agg,
    on=["poll_interval_id", "party_clean"],
    how="left",
    validate="one_to_one"
)

# Leere Partei-Intervalle sichtbar machen
party_interval["n_speeches"] = (
    party_interval["n_speeches"]
    .fillna(0)
    .astype(int)
)

party_interval["n_speeches_with_gal_tan"] = (
    party_interval["n_speeches_with_gal_tan"]
    .fillna(0)
    .astype(int)
)

for col in TOPIC_COUNT_COLS:
    party_interval[col] = (
        party_interval[col]
        .fillna(0)
        .astype(int)
    )

party_interval["is_mainstream_party"] = (
    party_interval["party_clean"].isin(MAINSTREAM_PARTIES)
)

party_interval["is_afd"] = (
    party_interval["party_clean"].eq("afd")
)

# =====================================================
# MODERATOR: ORDINALE NÄHE ZUR AFD
# =====================================================

afd_proximity_map = {
    "linke": 0,
    "grüne": 1,
    "spd": 2,
    "fdp": 3,
    "cdu/csu": 4
}

party_interval["afd_proximity_ordinal"] = (
    party_interval["party_clean"]
    .map(afd_proximity_map)
)

# Optional: Label zur besseren Lesbarkeit
afd_proximity_label_map = {
    0: "Die Linke",
    1: "Bündnis 90/Die Grünen",
    2: "SPD",
    3: "FDP",
    4: "CDU/CSU"
}

party_interval["afd_proximity_label"] = (
    party_interval["afd_proximity_ordinal"]
    .map(afd_proximity_label_map)
)

# =====================================================
# GAL-TAN-THEMENANTEILE INNERHALB DER PARTEI-INTERVALLE
# =====================================================

for topic, count_col in TOPIC_MAP.items():
    party_interval[f"share_{topic}_of_party_speeches"] = np.where(
        party_interval["n_speeches"] > 0,
        party_interval[count_col] / party_interval["n_speeches"],
        np.nan
    )

# =====================================================
# MAINSTREAM-GESAMTZAHLEN PRO INTERVALL
# =====================================================

mainstream_totals = (
    party_interval[
        party_interval["party_clean"].isin(MAINSTREAM_PARTIES)
    ]
    .groupby("poll_interval_id", as_index=False)
    .agg({
        "n_speeches": "sum",
        "n_speeches_environment": "sum",
        "n_speeches_migration": "sum",
        "n_speeches_society": "sum",
        "n_speeches_civil_rights": "sum",
        "n_speeches_cosmopolitanism": "sum"
    })
    .rename(columns={
        "n_speeches": "n_mainstream_speeches",
        "n_speeches_environment": "n_mainstream_speeches_environment",
        "n_speeches_migration": "n_mainstream_speeches_migration",
        "n_speeches_society": "n_mainstream_speeches_society",
        "n_speeches_civil_rights": "n_mainstream_speeches_civil_rights",
        "n_speeches_cosmopolitanism": "n_mainstream_speeches_cosmopolitanism"
    })
)

party_interval = party_interval.merge(
    mainstream_totals,
    on="poll_interval_id",
    how="left",
    validate="many_to_one"
)

# Anteil der Themenreden einer Partei an allen Mainstream-Reden im Intervall
for topic, count_col in TOPIC_MAP.items():
    party_interval[f"share_{topic}_of_mainstream_speeches"] = np.where(
        party_interval["n_mainstream_speeches"] > 0,
        party_interval[count_col] / party_interval["n_mainstream_speeches"],
        np.nan
    )

# Anteil einer Partei an allen Mainstream-Reden zu einem bestimmten Thema
for topic, count_col in TOPIC_MAP.items():
    mainstream_topic_col = f"n_mainstream_speeches_{topic}"

    party_interval[f"share_party_of_mainstream_{topic}_speeches"] = np.where(
        party_interval[mainstream_topic_col] > 0,
        party_interval[count_col] / party_interval[mainstream_topic_col],
        np.nan
    )

# =====================================================
# GOVERNMENT / OPPOSITION AUF INTERVALL-EBENE
# =====================================================

gov_interval = (
    speeches_with_polls
    .groupby(["poll_interval_id", "party_clean"])["government_label"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    .reset_index()
    .rename(columns={"government_label": "government_label_interval"})
)

party_interval = party_interval.merge(
    gov_interval,
    on=["poll_interval_id", "party_clean"],
    how="left",
    validate="one_to_one"
)

# Numerische Dummy-Variablen:
# 1 = ja, 0 = nein, NaN = keine Rede im Intervall / nicht bestimmbar
party_interval["is_government"] = np.where(
    party_interval["government_label_interval"].eq("Regierung"),
    1,
    np.where(
        party_interval["government_label_interval"].eq("Opposition"),
        0,
        np.nan
    )
)

party_interval["is_opposition"] = np.where(
    party_interval["government_label_interval"].eq("Opposition"),
    1,
    np.where(
        party_interval["government_label_interval"].eq("Regierung"),
        0,
        np.nan
    )
)

# =====================================================
# WAHLPERIODE AUF PARTEI × INTERVALL-EBENE
# =====================================================

wahlperiode_party_interval = (
    speeches_with_polls
    .groupby(["poll_interval_id", "party_clean"])["wahlperiode"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    .reset_index()
    .rename(columns={"wahlperiode": "wahlperiode"})
)

party_interval = party_interval.merge(
    wahlperiode_party_interval,
    on=["poll_interval_id", "party_clean"],
    how="left",
    validate="one_to_one"
)

party_interval["wahlperiode"] = party_interval["wahlperiode"].astype("Int64")
party_interval["wahlperiode_cat"] = (
    "WP" + party_interval["wahlperiode"].astype("string")
)

# =====================================================
# ZEITVARIABLEN
# =====================================================

party_interval["poll_year"] = party_interval["poll_date"].dt.year
party_interval["poll_month"] = party_interval["poll_date"].dt.month
party_interval["interval_length_days"] = (
    party_interval["next_poll_date"] -
    party_interval["poll_date"]
).dt.days

# Sortieren und speichern
party_interval = party_interval.sort_values(
    ["poll_interval_id", "party_clean"]
).reset_index(drop=True)

party_interval.to_csv(OUTPUT_PARTY_INTERVAL, index=False)

print(f"Gespeichert: {OUTPUT_PARTY_INTERVAL}")

# =====================================================
# VOLLSTÄNDIGES MAINSTREAM_ALL × POLLING-INTERVALL-GRID
# =====================================================

complete_mainstream_grid = poll_intervals[
    [
        "poll_interval_id",
        "poll_date",
        "next_poll_date",
        "afd_support_t"
    ]
].copy()

# =====================================================
# MAINSTREAM_ALL AGGREGIEREN
# =====================================================
# Redegewichtet: Parteien mit mehr Reden im Intervall fließen stärker ein.

mainstream_agg = (
    speeches_with_polls[
        speeches_with_polls["party_clean"].isin(MAINSTREAM_PARTIES)
    ]
    .groupby("poll_interval_id", as_index=False)
    .agg({
        # Indizes
        "populism_score": "mean",
        "populism_sum": "mean",
        "gal_tan_score": "mean",

        # Populismus-Subdimensionen
        "P1_people": "mean",
        "P2_anti_elite": "mean",
        "P3_outgroup": "mean",

        # GAL-TAN-Subdimensionen
        "G1_environment": "mean",
        "G2_migration": "mean",
        "G3_society": "mean",
        "G4_civil_rights": "mean",
        "G5_cosmopolitanism": "mean",

        # GAL-TAN-Themenhäufigkeiten
        "has_environment": "sum",
        "has_migration": "sum",
        "has_society": "sum",
        "has_civil_rights": "sum",
        "has_cosmopolitanism": "sum",

        # Fallzahlen / Checks
        "rede_id": "count",
        "has_gal_tan_score": "sum",
        "word_count": "mean"
    })
)

mainstream_agg = mainstream_agg.rename(columns={
    "populism_score": "mean_populism_score",
    "populism_sum": "mean_populism_sum",
    "gal_tan_score": "mean_gal_tan_score",

    "P1_people": "mean_P1_people",
    "P2_anti_elite": "mean_P2_anti_elite",
    "P3_outgroup": "mean_P3_outgroup",

    "G1_environment": "mean_G1_environment",
    "G2_migration": "mean_G2_migration",
    "G3_society": "mean_G3_society",
    "G4_civil_rights": "mean_G4_civil_rights",
    "G5_cosmopolitanism": "mean_G5_cosmopolitanism",

    "has_environment": "n_speeches_environment",
    "has_migration": "n_speeches_migration",
    "has_society": "n_speeches_society",
    "has_civil_rights": "n_speeches_civil_rights",
    "has_cosmopolitanism": "n_speeches_cosmopolitanism",

    "rede_id": "n_speeches",
    "has_gal_tan_score": "n_speeches_with_gal_tan",
    "word_count": "mean_word_count"
})

mainstream_interval = complete_mainstream_grid.merge(
    mainstream_agg,
    on="poll_interval_id",
    how="left",
    validate="one_to_one"
)

mainstream_interval["n_speeches"] = (
    mainstream_interval["n_speeches"]
    .fillna(0)
    .astype(int)
)

mainstream_interval["n_speeches_with_gal_tan"] = (
    mainstream_interval["n_speeches_with_gal_tan"]
    .fillna(0)
    .astype(int)
)

for col in TOPIC_COUNT_COLS:
    mainstream_interval[col] = (
        mainstream_interval[col]
        .fillna(0)
        .astype(int)
    )

for topic, count_col in TOPIC_MAP.items():
    mainstream_interval[f"share_{topic}_of_mainstream_speeches"] = np.where(
        mainstream_interval["n_speeches"] > 0,
        mainstream_interval[count_col] / mainstream_interval["n_speeches"],
        np.nan
    )

mainstream_interval["unit"] = "mainstream_all"

# =====================================================
# WAHLPERIODE AUF MAINSTREAM_ALL × INTERVALL-EBENE
# =====================================================

wahlperiode_mainstream_interval = (
    speeches_with_polls[
        speeches_with_polls["party_clean"].isin(MAINSTREAM_PARTIES)
    ]
    .groupby("poll_interval_id")["wahlperiode"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    .reset_index()
)

mainstream_interval = mainstream_interval.merge(
    wahlperiode_mainstream_interval,
    on="poll_interval_id",
    how="left",
    validate="one_to_one"
)

mainstream_interval["poll_year"] = mainstream_interval["poll_date"].dt.year
mainstream_interval["poll_month"] = mainstream_interval["poll_date"].dt.month
mainstream_interval["interval_length_days"] = (
    mainstream_interval["next_poll_date"] -
    mainstream_interval["poll_date"]
).dt.days

mainstream_interval = mainstream_interval.sort_values(
    "poll_interval_id"
).reset_index(drop=True)

mainstream_interval["wahlperiode"] = mainstream_interval["wahlperiode"].astype("Int64")
mainstream_interval["wahlperiode_cat"] = (
    "WP" + mainstream_interval["wahlperiode"].astype("string")
)

mainstream_interval.to_csv(OUTPUT_MAINSTREAM_INTERVAL, index=False)

print(f"Gespeichert: {OUTPUT_MAINSTREAM_INTERVAL}")

# =====================================================
# AFD-REFERENZWERTE PRO INTERVALL
# =====================================================
# Keine neue inhaltliche Aggregation:
# Die AfD-Zeile aus party_interval wird nur als Referenz für Distanzen genutzt.

afd_interval = party_interval[
    party_interval["party_clean"] == "afd"
][
    [
        "poll_interval_id",

        "mean_populism_score",
        "mean_populism_sum",
        "mean_gal_tan_score",

        "mean_P1_people",
        "mean_P2_anti_elite",
        "mean_P3_outgroup",

        "mean_G1_environment",
        "mean_G2_migration",
        "mean_G3_society",
        "mean_G4_civil_rights",
        "mean_G5_cosmopolitanism",

        "n_speeches",
        "n_speeches_with_gal_tan",

        "n_speeches_environment",
        "n_speeches_migration",
        "n_speeches_society",
        "n_speeches_civil_rights",
        "n_speeches_cosmopolitanism",

        "share_environment_of_party_speeches",
        "share_migration_of_party_speeches",
        "share_society_of_party_speeches",
        "share_civil_rights_of_party_speeches",
        "share_cosmopolitanism_of_party_speeches"
    ]
].copy()

afd_interval = afd_interval.rename(columns={
    "mean_populism_score": "afd_mean_populism_score",
    "mean_populism_sum": "afd_mean_populism_sum",
    "mean_gal_tan_score": "afd_mean_gal_tan_score",

    "mean_P1_people": "afd_mean_P1_people",
    "mean_P2_anti_elite": "afd_mean_P2_anti_elite",
    "mean_P3_outgroup": "afd_mean_P3_outgroup",

    "mean_G1_environment": "afd_mean_G1_environment",
    "mean_G2_migration": "afd_mean_G2_migration",
    "mean_G3_society": "afd_mean_G3_society",
    "mean_G4_civil_rights": "afd_mean_G4_civil_rights",
    "mean_G5_cosmopolitanism": "afd_mean_G5_cosmopolitanism",

    "n_speeches": "afd_n_speeches",
    "n_speeches_with_gal_tan": "afd_n_speeches_with_gal_tan",

    "n_speeches_environment": "afd_n_speeches_environment",
    "n_speeches_migration": "afd_n_speeches_migration",
    "n_speeches_society": "afd_n_speeches_society",
    "n_speeches_civil_rights": "afd_n_speeches_civil_rights",
    "n_speeches_cosmopolitanism": "afd_n_speeches_cosmopolitanism",

    "share_environment_of_party_speeches": "afd_share_environment_of_party_speeches",
    "share_migration_of_party_speeches": "afd_share_migration_of_party_speeches",
    "share_society_of_party_speeches": "afd_share_society_of_party_speeches",
    "share_civil_rights_of_party_speeches": "afd_share_civil_rights_of_party_speeches",
    "share_cosmopolitanism_of_party_speeches": "afd_share_cosmopolitanism_of_party_speeches"
})

# =====================================================
# DISTANZEN: MAINSTREAM-PARTEIEN ZUR AFD
# =====================================================

party_distance = party_interval.merge(
    afd_interval,
    on="poll_interval_id",
    how="left",
    validate="many_to_one"
)

# Für Hypothesentests nur Mainstream-Parteien
party_distance = party_distance[
    party_distance["party_clean"].isin(MAINSTREAM_PARTIES)
].copy()

# Zentrale Distanzen
party_distance["populism_distance_to_afd"] = (
    party_distance["mean_populism_score"] -
    party_distance["afd_mean_populism_score"]
).abs()

party_distance["populism_sum_distance_to_afd"] = (
    party_distance["mean_populism_sum"] -
    party_distance["afd_mean_populism_sum"]
).abs()

party_distance["gal_tan_distance_to_afd"] = (
    party_distance["mean_gal_tan_score"] -
    party_distance["afd_mean_gal_tan_score"]
).abs()

# Populismus-Subdimensionen
party_distance["P1_people_distance_to_afd"] = (
    party_distance["mean_P1_people"] -
    party_distance["afd_mean_P1_people"]
).abs()

party_distance["P2_anti_elite_distance_to_afd"] = (
    party_distance["mean_P2_anti_elite"] -
    party_distance["afd_mean_P2_anti_elite"]
).abs()

party_distance["P3_outgroup_distance_to_afd"] = (
    party_distance["mean_P3_outgroup"] -
    party_distance["afd_mean_P3_outgroup"]
).abs()

# GAL-TAN-Subdimensionen
party_distance["G1_environment_distance_to_afd"] = (
    party_distance["mean_G1_environment"] -
    party_distance["afd_mean_G1_environment"]
).abs()

party_distance["G2_migration_distance_to_afd"] = (
    party_distance["mean_G2_migration"] -
    party_distance["afd_mean_G2_migration"]
).abs()

party_distance["G3_society_distance_to_afd"] = (
    party_distance["mean_G3_society"] -
    party_distance["afd_mean_G3_society"]
).abs()

party_distance["G4_civil_rights_distance_to_afd"] = (
    party_distance["mean_G4_civil_rights"] -
    party_distance["afd_mean_G4_civil_rights"]
).abs()

party_distance["G5_cosmopolitanism_distance_to_afd"] = (
    party_distance["mean_G5_cosmopolitanism"] -
    party_distance["afd_mean_G5_cosmopolitanism"]
).abs()

party_distance = party_distance.sort_values(
    ["poll_interval_id", "party_clean"]
).reset_index(drop=True)

party_distance.to_csv(OUTPUT_PARTY_DISTANCE, index=False)

print(f"Gespeichert: {OUTPUT_PARTY_DISTANCE}")

# =====================================================
# DISTANZEN: MAINSTREAM_ALL ZUR AFD
# =====================================================

mainstream_distance = mainstream_interval.merge(
    afd_interval,
    on="poll_interval_id",
    how="left",
    validate="one_to_one"
)

mainstream_distance["populism_distance_to_afd"] = (
    mainstream_distance["mean_populism_score"] -
    mainstream_distance["afd_mean_populism_score"]
).abs()

mainstream_distance["populism_sum_distance_to_afd"] = (
    mainstream_distance["mean_populism_sum"] -
    mainstream_distance["afd_mean_populism_sum"]
).abs()

mainstream_distance["gal_tan_distance_to_afd"] = (
    mainstream_distance["mean_gal_tan_score"] -
    mainstream_distance["afd_mean_gal_tan_score"]
).abs()

mainstream_distance["P1_people_distance_to_afd"] = (
    mainstream_distance["mean_P1_people"] -
    mainstream_distance["afd_mean_P1_people"]
).abs()

mainstream_distance["P2_anti_elite_distance_to_afd"] = (
    mainstream_distance["mean_P2_anti_elite"] -
    mainstream_distance["afd_mean_P2_anti_elite"]
).abs()

mainstream_distance["P3_outgroup_distance_to_afd"] = (
    mainstream_distance["mean_P3_outgroup"] -
    mainstream_distance["afd_mean_P3_outgroup"]
).abs()

mainstream_distance["G1_environment_distance_to_afd"] = (
    mainstream_distance["mean_G1_environment"] -
    mainstream_distance["afd_mean_G1_environment"]
).abs()

mainstream_distance["G2_migration_distance_to_afd"] = (
    mainstream_distance["mean_G2_migration"] -
    mainstream_distance["afd_mean_G2_migration"]
).abs()

mainstream_distance["G3_society_distance_to_afd"] = (
    mainstream_distance["mean_G3_society"] -
    mainstream_distance["afd_mean_G3_society"]
).abs()

mainstream_distance["G4_civil_rights_distance_to_afd"] = (
    mainstream_distance["mean_G4_civil_rights"] -
    mainstream_distance["afd_mean_G4_civil_rights"]
).abs()

mainstream_distance["G5_cosmopolitanism_distance_to_afd"] = (
    mainstream_distance["mean_G5_cosmopolitanism"] -
    mainstream_distance["afd_mean_G5_cosmopolitanism"]
).abs()

mainstream_distance = mainstream_distance.sort_values(
    "poll_interval_id"
).reset_index(drop=True)

mainstream_distance.to_csv(OUTPUT_MAINSTREAM_DISTANCE, index=False)

print(f"Gespeichert: {OUTPUT_MAINSTREAM_DISTANCE}")

# =====================================================
# KONTROLLAUSGABEN
# =====================================================

print("\n=====================================================")
print("KONTROLLEN")
print("=====================================================")

print("\nSpeech-Level mit Polling:")
print(speeches_with_polls.shape)

print("\nPartei × Polling-Intervall:")
print(party_interval.shape)
print(party_interval["party_clean"].value_counts())

print("\nMainstream × Polling-Intervall:")
print(mainstream_interval.shape)

print("\nPartei-Distanzen:")
print(party_distance.shape)
print(party_distance["party_clean"].value_counts())

print("\nMainstream-Distanzen:")
print(mainstream_distance.shape)

print("\nZeitraum der gematchten Reden:")
print(
    speeches_with_polls["sitzung_datum"].min(),
    "bis",
    speeches_with_polls["sitzung_datum"].max()
)

print("\nVollständige Polling-Intervalle:")
print("Anzahl poll_intervals:", len(poll_intervals))
print("Erwartete Partei-Intervall-Zeilen:", len(poll_intervals) * len(ALL_PARTIES))
print("Tatsächliche Partei-Intervall-Zeilen:", len(party_interval))

print("\nLeere Partei-Intervalle ohne Reden:")
print(
    party_interval
    .groupby("party_clean")["n_speeches"]
    .apply(lambda x: (x == 0).sum())
)

print("\nReden pro Partei insgesamt über alle Intervalle:")
print(
    party_interval
    .groupby("party_clean")["n_speeches"]
    .sum()
    .sort_values(ascending=False)
)

print("\nFehlende party_support_t nach Partei:")
print(
    party_interval
    .groupby("party_clean")["party_support_t"]
    .apply(lambda x: x.isna().sum())
)

print("\nFehlende party_support_change nach Partei:")
print(
    party_interval
    .groupby("party_clean")["party_support_change"]
    .apply(lambda x: x.isna().sum())
)

print("\nSupport Change, deskriptiv für Mainstream-Parteien:")
print(
    party_interval[
        party_interval["party_clean"].isin(MAINSTREAM_PARTIES)
    ]["party_support_change"].describe()
)

print("\nThemenhäufigkeiten über alle Partei-Intervalle:")
print(
    party_interval[
        [
            "n_speeches_environment",
            "n_speeches_migration",
            "n_speeches_society",
            "n_speeches_civil_rights",
            "n_speeches_cosmopolitanism"
        ]
    ].sum()
)

print("\nAfD-Intervalle ohne Reden:")
print(
    party_interval[
        party_interval["party_clean"].eq("afd")
    ]["n_speeches"].eq(0).sum()
)

print("\nDistanzen mit fehlendem AfD-Referenzwert:")
print(
    party_distance[
        [
            "afd_mean_populism_score",
            "afd_mean_gal_tan_score",
            "populism_distance_to_afd",
            "gal_tan_distance_to_afd"
        ]
    ].isna().sum()
)

print("\nBeispiel: Partei-Distanz-Datei")
print(party_distance.head())

print("\nFertig.")