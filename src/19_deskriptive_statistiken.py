from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. EINSTELLUNGEN
# ============================================================

INPUT_SPEECH = Path(
    "data/processed/final_analysis/"
    "02_speech_level_with_polling.csv"
)

INPUT_PARTY_INTERVAL = Path(
    "data/processed/final_analysis/"
    "05_party_polling_interval_distances.csv"
)

INPUT_MAINSTREAM_INTERVAL = Path(
    "data/processed/final_analysis/"
    "06_mainstream_polling_interval_distances.csv"
)

OUTPUT_DIR = Path(
    "data/processed/descriptives"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DECIMALS = 3

PARTY_ORDER = [
    "linke",
    "grüne",
    "spd",
    "fdp",
    "cdu/csu",
    "afd",
]

MAINSTREAM_PARTIES = [
    "linke",
    "grüne",
    "spd",
    "fdp",
    "cdu/csu",
]


# ============================================================
# 2. VARIABLENDEFINITIONEN
# ============================================================

POPULISM_SCORE = "populism_score"

POPULISM_SUBDIMENSIONS = [
    "P1_people",
    "P2_anti_elite",
    "P3_outgroup",
]

GAL_TAN_SCORE = "gal_tan_score"

GAL_TAN_SUBDIMENSIONS = [
    "G1_environment",
    "G2_migration",
    "G3_society",
    "G4_civil_rights",
    "G5_cosmopolitanism",
]

ALL_SCORE_VARIABLES = (
    [POPULISM_SCORE]
    + POPULISM_SUBDIMENSIONS
    + [GAL_TAN_SCORE]
    + GAL_TAN_SUBDIMENSIONS
)

VARIABLE_LABELS = {
    "populism_score": "Populismus-Gesamtscore",
    "P1_people": "Volkszentrierung",
    "P2_anti_elite": "Anti-Elitismus",
    "P3_outgroup": "Ausschluss von Outgroups",

    "gal_tan_score": "GAL-TAN-Gesamtscore",
    "G1_environment": "Umwelt",
    "G2_migration": "Migration",
    "G3_society": "Gesellschaftspolitik",
    "G4_civil_rights": "Bürgerrechte",
    "G5_cosmopolitanism": "Kosmopolitismus",
}

VARIABLE_GROUPS = {
    "populism_score": "Populismus-Gesamtscore",
    "P1_people": "Populismus-Subdimension",
    "P2_anti_elite": "Populismus-Subdimension",
    "P3_outgroup": "Populismus-Subdimension",

    "gal_tan_score": "GAL-TAN-Gesamtscore",
    "G1_environment": "GAL-TAN-Subdimension",
    "G2_migration": "GAL-TAN-Subdimension",
    "G3_society": "GAL-TAN-Subdimension",
    "G4_civil_rights": "GAL-TAN-Subdimension",
    "G5_cosmopolitanism": "GAL-TAN-Subdimension",
}

VARIABLE_CONSTRUCTS = {
    "populism_score": "Populismus",
    "P1_people": "Populismus",
    "P2_anti_elite": "Populismus",
    "P3_outgroup": "Populismus",

    "gal_tan_score": "GAL-TAN",
    "G1_environment": "GAL-TAN",
    "G2_migration": "GAL-TAN",
    "G3_society": "GAL-TAN",
    "G4_civil_rights": "GAL-TAN",
    "G5_cosmopolitanism": "GAL-TAN",
}


# ============================================================
# 3. GAL-TAN-THEMENINDIKATOREN AUF REDEEBENE
# ============================================================

GAL_TAN_TOPIC_INDICATORS = {
    "G1_environment": "has_environment",
    "G2_migration": "has_migration",
    "G3_society": "has_society",
    "G4_civil_rights": "has_civil_rights",
    "G5_cosmopolitanism": "has_cosmopolitanism",
}


# ============================================================
# 4. VARIABLEN AUF INTERVALL-EBENE
# ============================================================

INTERVAL_OVERALL_VARIABLES = {
    "Populismus": {
        "mainstream_absolute": "mean_populism_score",
        "afd_absolute": "afd_mean_populism_score",
        "distance_to_afd": "populism_distance_to_afd",
    },
    "GAL-TAN": {
        "mainstream_absolute": "mean_gal_tan_score",
        "afd_absolute": "afd_mean_gal_tan_score",
        "distance_to_afd": "gal_tan_distance_to_afd",
    },
}

INTERVAL_SUBDIMENSION_VARIABLES = {
    "P1_people": {
        "mainstream_absolute": "mean_P1_people",
        "afd_absolute": "afd_mean_P1_people",
        "distance_to_afd": "P1_people_distance_to_afd",
    },
    "P2_anti_elite": {
        "mainstream_absolute": "mean_P2_anti_elite",
        "afd_absolute": "afd_mean_P2_anti_elite",
        "distance_to_afd": "P2_anti_elite_distance_to_afd",
    },
    "P3_outgroup": {
        "mainstream_absolute": "mean_P3_outgroup",
        "afd_absolute": "afd_mean_P3_outgroup",
        "distance_to_afd": "P3_outgroup_distance_to_afd",
    },

    "G1_environment": {
        "mainstream_absolute": "mean_G1_environment",
        "afd_absolute": "afd_mean_G1_environment",
        "distance_to_afd": "G1_environment_distance_to_afd",
    },
    "G2_migration": {
        "mainstream_absolute": "mean_G2_migration",
        "afd_absolute": "afd_mean_G2_migration",
        "distance_to_afd": "G2_migration_distance_to_afd",
    },
    "G3_society": {
        "mainstream_absolute": "mean_G3_society",
        "afd_absolute": "afd_mean_G3_society",
        "distance_to_afd": "G3_society_distance_to_afd",
    },
    "G4_civil_rights": {
        "mainstream_absolute": "mean_G4_civil_rights",
        "afd_absolute": "afd_mean_G4_civil_rights",
        "distance_to_afd": "G4_civil_rights_distance_to_afd",
    },
    "G5_cosmopolitanism": {
        "mainstream_absolute": "mean_G5_cosmopolitanism",
        "afd_absolute": "afd_mean_G5_cosmopolitanism",
        "distance_to_afd": "G5_cosmopolitanism_distance_to_afd",
    },
}


# ============================================================
# 5. HILFSFUNKTIONEN
# ============================================================

def check_file_exists(filepath):
    """
    Prüft, ob eine Eingabedatei existiert.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden:\n{filepath.resolve()}"
        )


def check_required_columns(
    data,
    required_columns,
    dataset_name,
):
    """
    Prüft, ob alle benötigten Variablen vorhanden sind.
    """
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        missing_text = "\n".join(
            f"- {column}"
            for column in missing_columns
        )

        raise KeyError(
            f"Im Datensatz '{dataset_name}' fehlen "
            f"folgende Variablen:\n{missing_text}"
        )


def standardize_party(series):
    """
    Standardisiert die Parteibezeichnungen.
    """
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def descriptive_stats_for_series(series):
    """
    Berechnet deskriptive Statistiken für eine Variable.
    """
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return pd.Series({
        "n_total": int(numeric.size),
        "n_valid": int(numeric.count()),
        "n_missing": int(numeric.isna().sum()),
        "share_missing": numeric.isna().mean(),
        "percent_missing": numeric.isna().mean() * 100,
        "mean": numeric.mean(),
        "sd": numeric.std(ddof=1),
        "min": numeric.min(),
        "p25": numeric.quantile(0.25),
        "median": numeric.median(),
        "p75": numeric.quantile(0.75),
        "max": numeric.max(),
    })


def descriptive_stats_overall(
    data,
    variable,
    group_label="gesamt",
):
    """
    Berechnet deskriptive Statistiken insgesamt.
    """
    stats = descriptive_stats_for_series(
        data[variable]
    ).to_frame().T

    stats.insert(
        0,
        "group",
        group_label,
    )

    stats.insert(
        1,
        "construct",
        VARIABLE_CONSTRUCTS[variable],
    )

    stats.insert(
        2,
        "variable",
        variable,
    )

    stats.insert(
        3,
        "variable_label",
        VARIABLE_LABELS[variable],
    )

    stats.insert(
        4,
        "variable_group",
        VARIABLE_GROUPS[variable],
    )

    return stats


def descriptive_stats_by_party(
    data,
    variable,
):
    """
    Berechnet deskriptive Statistiken pro Partei.
    """
    output = (
        data
        .groupby(
            "party_clean",
            observed=False,
        )[variable]
        .apply(
            descriptive_stats_for_series
        )
        .unstack()
        .reset_index()
    )

    output.insert(
        1,
        "construct",
        VARIABLE_CONSTRUCTS[variable],
    )

    output.insert(
        2,
        "variable",
        variable,
    )

    output.insert(
        3,
        "variable_label",
        VARIABLE_LABELS[variable],
    )

    output.insert(
        4,
        "variable_group",
        VARIABLE_GROUPS[variable],
    )

    return output


def create_party_rankings(
    data,
    variables,
):
    """
    Erstellt Mittelwerte, Standardabweichungen und Rangfolgen
    der Parteien für mehrere Variablen.
    """
    tables = []

    for variable in variables:
        grouped = (
            data
            .groupby(
                "party_clean",
                observed=False,
            )[variable]
            .agg(
                n_valid="count",
                mean="mean",
                sd="std",
                median="median",
                min="min",
                max="max",
            )
            .reset_index()
        )

        grouped.insert(
            1,
            "construct",
            VARIABLE_CONSTRUCTS[variable],
        )

        grouped.insert(
            2,
            "variable",
            variable,
        )

        grouped.insert(
            3,
            "variable_label",
            VARIABLE_LABELS[variable],
        )

        grouped["rank_highest_value"] = (
            grouped["mean"]
            .rank(
                method="min",
                ascending=False,
            )
            .astype("Int64")
        )

        grouped["rank_lowest_value"] = (
            grouped["mean"]
            .rank(
                method="min",
                ascending=True,
            )
            .astype("Int64")
        )

        tables.append(grouped)

    output = pd.concat(
        tables,
        ignore_index=True,
    )

    output["party_clean"] = pd.Categorical(
        output["party_clean"],
        categories=PARTY_ORDER,
        ordered=True,
    )

    return (
        output
        .sort_values(
            [
                "construct",
                "variable",
                "rank_highest_value",
                "party_clean",
            ]
        )
        .reset_index(drop=True)
    )


def create_mainstream_speech_stats(
    data,
    variables,
):
    """
    Berechnet redegewichtete Werte der Mainstream-Parteien.
    """
    mainstream_data = data.loc[
        data["party_clean"].isin(
            MAINSTREAM_PARTIES
        )
    ].copy()

    rows = []

    for variable in variables:
        stats = descriptive_stats_for_series(
            mainstream_data[variable]
        ).to_dict()

        rows.append({
            "group": "Mainstream-Parteien",
            "weighting": "redegewichtet",
            "construct": VARIABLE_CONSTRUCTS[variable],
            "variable": variable,
            "variable_label": VARIABLE_LABELS[variable],
            **stats,
        })

    return pd.DataFrame(rows)


def create_party_balanced_mainstream_stats(
    data,
    variables,
):
    """
    Berechnet parteigewichtete Mainstream-Werte.

    Jede Mainstream-Partei geht mit demselben Gewicht ein.
    """
    mainstream_data = data.loc[
        data["party_clean"].isin(
            MAINSTREAM_PARTIES
        )
    ].copy()

    rows = []

    for variable in variables:
        party_means = (
            mainstream_data
            .groupby(
                "party_clean",
                observed=True,
            )[variable]
            .mean()
            .dropna()
        )

        rows.append({
            "group": "Mainstream-Parteien",
            "weighting": "parteigewichtet",
            "construct": VARIABLE_CONSTRUCTS[variable],
            "variable": variable,
            "variable_label": VARIABLE_LABELS[variable],
            "n_parties": int(party_means.count()),
            "mean": party_means.mean(),
            "sd_between_parties": party_means.std(ddof=1),
            "min_party_mean": party_means.min(),
            "median_party_mean": party_means.median(),
            "max_party_mean": party_means.max(),
        })

    return pd.DataFrame(rows)


def create_interval_overall_stats(
    data,
    variable_mapping,
    analysis_level,
):
    """
    Berechnet absolute Mainstream-Werte, absolute AfD-Werte
    und Distanzen zur AfD.
    """
    rows = []

    for construct, mapping in variable_mapping.items():
        for measure_type, variable in mapping.items():
            stats = descriptive_stats_for_series(
                data[variable]
            ).to_dict()

            rows.append({
                "analysis_level": analysis_level,
                "construct": construct,
                "measure_type": measure_type,
                "variable": variable,
                **stats,
            })

    return pd.DataFrame(rows)


def create_interval_subdimension_stats(
    data,
    variable_mapping,
    analysis_level,
):
    """
    Berechnet parallele Statistiken für alle Populismus-
    und GAL-TAN-Subdimensionen.
    """
    rows = []

    for subdimension, mapping in variable_mapping.items():
        construct = VARIABLE_CONSTRUCTS[subdimension]

        for measure_type, variable in mapping.items():
            stats = descriptive_stats_for_series(
                data[variable]
            ).to_dict()

            rows.append({
                "analysis_level": analysis_level,
                "construct": construct,
                "subdimension": subdimension,
                "subdimension_label": (
                    VARIABLE_LABELS[subdimension]
                ),
                "measure_type": measure_type,
                "variable": variable,
                **stats,
            })

    return pd.DataFrame(rows)


def create_interval_overall_by_party(
    data,
    variable_mapping,
):
    """
    Berechnet absolute Werte und Distanzen auf
    Partei-Intervall-Ebene getrennt nach Partei.

    Die AfD-Werte werden nicht nach Mainstream-Partei
    ausgewiesen, da sie innerhalb eines Intervalls
    für alle Mainstream-Parteien identisch sind.
    """
    rows = []

    for construct, mapping in variable_mapping.items():
        for measure_type in [
            "mainstream_absolute",
            "distance_to_afd",
        ]:
            variable = mapping[measure_type]

            grouped = (
                data
                .groupby(
                    "party_clean",
                    observed=False,
                )[variable]
                .agg(
                    n_valid="count",
                    mean="mean",
                    sd="std",
                    median="median",
                    min="min",
                    max="max",
                )
                .reset_index()
            )

            grouped.insert(
                1,
                "construct",
                construct,
            )

            grouped.insert(
                2,
                "measure_type",
                measure_type,
            )

            grouped.insert(
                3,
                "variable",
                variable,
            )

            grouped["rank_highest_value"] = (
                grouped["mean"]
                .rank(
                    method="min",
                    ascending=False,
                )
                .astype("Int64")
            )

            grouped["rank_lowest_value"] = (
                grouped["mean"]
                .rank(
                    method="min",
                    ascending=True,
                )
                .astype("Int64")
            )

            rows.append(grouped)

    return pd.concat(
        rows,
        ignore_index=True,
    )


def create_interval_subdimensions_by_party(
    data,
    variable_mapping,
):
    """
    Berechnet absolute Subdimensionswerte und Distanzen
    getrennt nach Mainstream-Partei.
    """
    rows = []

    for subdimension, mapping in variable_mapping.items():
        construct = VARIABLE_CONSTRUCTS[subdimension]

        for measure_type in [
            "mainstream_absolute",
            "distance_to_afd",
        ]:
            variable = mapping[measure_type]

            grouped = (
                data
                .groupby(
                    "party_clean",
                    observed=False,
                )[variable]
                .agg(
                    n_valid="count",
                    mean="mean",
                    sd="std",
                    median="median",
                    min="min",
                    max="max",
                )
                .reset_index()
            )

            grouped.insert(
                1,
                "construct",
                construct,
            )

            grouped.insert(
                2,
                "subdimension",
                subdimension,
            )

            grouped.insert(
                3,
                "subdimension_label",
                VARIABLE_LABELS[subdimension],
            )

            grouped.insert(
                4,
                "measure_type",
                measure_type,
            )

            grouped.insert(
                5,
                "variable",
                variable,
            )

            grouped["rank_highest_value"] = (
                grouped["mean"]
                .rank(
                    method="min",
                    ascending=False,
                )
                .astype("Int64")
            )

            grouped["rank_lowest_value"] = (
                grouped["mean"]
                .rank(
                    method="min",
                    ascending=True,
                )
                .astype("Int64")
            )

            rows.append(grouped)

    return pd.concat(
        rows,
        ignore_index=True,
    )


def create_support_change_stats(data):
    """
    Berechnet die Unterstützungsveränderung.

    party_support_change =
    aktueller Umfragewert minus vorheriger Umfragewert.

    Negative Werte entsprechen Verlusten.
    Positive Werte entsprechen Gewinnen.
    """
    values = pd.to_numeric(
        data["party_support_change"],
        errors="coerce",
    )

    descriptives = (
        descriptive_stats_for_series(values)
        .to_frame()
        .T
    )

    descriptives.insert(
        0,
        "variable",
        "party_support_change",
    )

    valid = pd.DataFrame({
        "party_support_change": values
    }).dropna()

    tolerance = 1e-10

    valid["change_category"] = np.select(
        [
            valid["party_support_change"] < -tolerance,
            valid["party_support_change"].abs() <= tolerance,
            valid["party_support_change"] > tolerance,
        ],
        [
            "Verlust",
            "Unverändert",
            "Gewinn",
        ],
        default="Unklar",
    )

    categories = (
        valid
        .groupby(
            "change_category",
            observed=True,
        )
        .size()
        .reset_index(name="n")
    )

    categories["share"] = (
        categories["n"]
        / categories["n"].sum()
    )

    categories["percent"] = (
        categories["share"] * 100
    )

    category_order = [
        "Verlust",
        "Unverändert",
        "Gewinn",
        "Unklar",
    ]

    categories["change_category"] = pd.Categorical(
        categories["change_category"],
        categories=category_order,
        ordered=True,
    )

    categories = (
        categories
        .sort_values("change_category")
        .reset_index(drop=True)
    )

    return descriptives, categories


def safe_round(
    dataframe,
    decimals=DECIMALS,
):
    """
    Rundet alle numerischen Spalten.
    """
    output = dataframe.copy()

    numeric_columns = output.select_dtypes(
        include=[np.number]
    ).columns

    output[numeric_columns] = (
        output[numeric_columns]
        .round(decimals)
    )

    return output


def write_csv(
    dataframe,
    filename,
):
    """
    Speichert einen DataFrame als CSV-Datei.

    - Semikolon als Spaltentrennzeichen
    - Punkt als Dezimaltrennzeichen
    - Fließkommazahlen immer mit drei Nachkommastellen
    """
    if dataframe is None or dataframe.empty:
        return

    output_path = OUTPUT_DIR / f"{filename}.csv"

    dataframe.to_csv(
        output_path,
        index=False,
        sep=";",
        decimal=".",
        encoding="utf-8-sig",
        float_format="%.3f",
    )


# ============================================================
# 6. DATEIEN PRÜFEN UND LADEN
# ============================================================

check_file_exists(INPUT_SPEECH)
check_file_exists(INPUT_PARTY_INTERVAL)
check_file_exists(INPUT_MAINSTREAM_INTERVAL)

speech_df = pd.read_csv(
    INPUT_SPEECH,
    low_memory=False,
)

party_interval_df = pd.read_csv(
    INPUT_PARTY_INTERVAL,
    low_memory=False,
)

mainstream_interval_df = pd.read_csv(
    INPUT_MAINSTREAM_INTERVAL,
    low_memory=False,
)

print("=" * 78)
print("DESKRIPTIVE STATISTIKEN")
print("=" * 78)

print(
    f"\nRedeebene: {len(speech_df)} Zeilen, "
    f"{len(speech_df.columns)} Spalten"
)

print(
    f"Partei-Intervall-Ebene: "
    f"{len(party_interval_df)} Zeilen, "
    f"{len(party_interval_df.columns)} Spalten"
)

print(
    f"Aggregierte Mainstream-Ebene: "
    f"{len(mainstream_interval_df)} Zeilen, "
    f"{len(mainstream_interval_df.columns)} Spalten"
)


# ============================================================
# 7. ERFORDERLICHE VARIABLEN PRÜFEN
# ============================================================

required_speech_columns = [
    "rede_id",
    "party_clean",
    "wahlperiode",
    "word_count",

    "populism_score",
    "P1_people",
    "P2_anti_elite",
    "P3_outgroup",

    "gal_tan_score",
    "G1_environment",
    "G2_migration",
    "G3_society",
    "G4_civil_rights",
    "G5_cosmopolitanism",

    "has_environment",
    "has_migration",
    "has_society",
    "has_civil_rights",
    "has_cosmopolitanism",
]

check_required_columns(
    data=speech_df,
    required_columns=required_speech_columns,
    dataset_name="02_speech_level_with_polling.csv",
)

required_party_interval_columns = [
    "poll_interval_id",
    "party_clean",
    "is_mainstream_party",
    "party_support_change",
    "party_support_change_abs",

    "mean_populism_score",
    "afd_mean_populism_score",
    "populism_distance_to_afd",

    "mean_gal_tan_score",
    "afd_mean_gal_tan_score",
    "gal_tan_distance_to_afd",
]

for mapping in INTERVAL_SUBDIMENSION_VARIABLES.values():
    required_party_interval_columns.extend(
        mapping.values()
    )

required_party_interval_columns = list(
    dict.fromkeys(
        required_party_interval_columns
    )
)

check_required_columns(
    data=party_interval_df,
    required_columns=required_party_interval_columns,
    dataset_name=(
        "05_party_polling_interval_distances.csv"
    ),
)

required_mainstream_interval_columns = [
    "poll_interval_id",

    "mean_populism_score",
    "afd_mean_populism_score",
    "populism_distance_to_afd",

    "mean_gal_tan_score",
    "afd_mean_gal_tan_score",
    "gal_tan_distance_to_afd",
]

for mapping in INTERVAL_SUBDIMENSION_VARIABLES.values():
    required_mainstream_interval_columns.extend(
        mapping.values()
    )

required_mainstream_interval_columns = list(
    dict.fromkeys(
        required_mainstream_interval_columns
    )
)

check_required_columns(
    data=mainstream_interval_df,
    required_columns=required_mainstream_interval_columns,
    dataset_name=(
        "06_mainstream_polling_interval_distances.csv"
    ),
)


# ============================================================
# 8. REDE-DATENSATZ BEREINIGEN
# ============================================================

speech_df["party_clean"] = standardize_party(
    speech_df["party_clean"]
)

speech_df = speech_df.loc[
    speech_df["party_clean"].isin(
        PARTY_ORDER
    )
].copy()

speech_numeric_columns = (
    ALL_SCORE_VARIABLES
    + [
        "word_count",
        "wahlperiode",
        "has_environment",
        "has_migration",
        "has_society",
        "has_civil_rights",
        "has_cosmopolitanism",
    ]
)

for column in speech_numeric_columns:
    speech_df[column] = pd.to_numeric(
        speech_df[column],
        errors="coerce",
    )

speech_df["party_clean"] = pd.Categorical(
    speech_df["party_clean"],
    categories=PARTY_ORDER,
    ordered=True,
)


# ============================================================
# 9. PARTEI-INTERVALL-DATENSATZ BEREINIGEN
# ============================================================

party_interval_df["party_clean"] = standardize_party(
    party_interval_df["party_clean"]
)

party_interval_df["is_mainstream_party"] = (
    pd.to_numeric(
        party_interval_df["is_mainstream_party"],
        errors="coerce",
    )
)

party_interval_df = party_interval_df.loc[
    (
        party_interval_df["is_mainstream_party"] == 1
    )
    & (
        party_interval_df["party_clean"].isin(
            MAINSTREAM_PARTIES
        )
    )
].copy()

party_interval_df["party_clean"] = pd.Categorical(
    party_interval_df["party_clean"],
    categories=MAINSTREAM_PARTIES,
    ordered=True,
)

party_interval_numeric_columns = [
    "party_support_change",
    "party_support_change_abs",

    "mean_populism_score",
    "afd_mean_populism_score",
    "populism_distance_to_afd",

    "mean_gal_tan_score",
    "afd_mean_gal_tan_score",
    "gal_tan_distance_to_afd",
]

for mapping in INTERVAL_SUBDIMENSION_VARIABLES.values():
    party_interval_numeric_columns.extend(
        mapping.values()
    )

party_interval_numeric_columns = list(
    dict.fromkeys(
        party_interval_numeric_columns
    )
)

for column in party_interval_numeric_columns:
    party_interval_df[column] = pd.to_numeric(
        party_interval_df[column],
        errors="coerce",
    )


# ============================================================
# 10. AGGREGIERTEN MAINSTREAM-DATENSATZ BEREINIGEN
# ============================================================

mainstream_interval_numeric_columns = [
    "mean_populism_score",
    "afd_mean_populism_score",
    "populism_distance_to_afd",

    "mean_gal_tan_score",
    "afd_mean_gal_tan_score",
    "gal_tan_distance_to_afd",
]

for mapping in INTERVAL_SUBDIMENSION_VARIABLES.values():
    mainstream_interval_numeric_columns.extend(
        mapping.values()
    )

mainstream_interval_numeric_columns = list(
    dict.fromkeys(
        mainstream_interval_numeric_columns
    )
)

for column in mainstream_interval_numeric_columns:
    mainstream_interval_df[column] = pd.to_numeric(
        mainstream_interval_df[column],
        errors="coerce",
    )


# ============================================================
# 11. REDEEBENE:
#     DESKRIPTIVE STATISTIKEN INSGESAMT
# ============================================================

overall_tables = []

for variable in ALL_SCORE_VARIABLES:
    overall_tables.append(
        descriptive_stats_overall(
            data=speech_df,
            variable=variable,
        )
    )

speech_scores_overall = pd.concat(
    overall_tables,
    ignore_index=True,
)

speech_scores_overall = (
    speech_scores_overall
    .sort_values(
        [
            "construct",
            "variable_group",
            "variable",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 12. REDEEBENE:
#     DESKRIPTIVE STATISTIKEN NACH PARTEI
# ============================================================

party_tables = []

for variable in ALL_SCORE_VARIABLES:
    party_tables.append(
        descriptive_stats_by_party(
            data=speech_df,
            variable=variable,
        )
    )

speech_scores_by_party = pd.concat(
    party_tables,
    ignore_index=True,
)

speech_scores_by_party = (
    speech_scores_by_party
    .sort_values(
        [
            "construct",
            "variable_group",
            "variable",
            "party_clean",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 13. POPULISMUS-PRÄVALENZ AUF REDEEBENE
#
# Anteil der gültigen Codierungen mit Score 1 oder 2.
# ============================================================

populism_prevalence_rows = []

for variable in POPULISM_SUBDIMENSIONS:
    values = pd.to_numeric(
        speech_df[variable],
        errors="coerce",
    )

    valid = values.dropna()

    score_1_or_2 = valid.isin(
        [1, 2]
    )

    populism_prevalence_rows.append({
        "construct": "Populismus",
        "variable": variable,
        "variable_label": VARIABLE_LABELS[variable],
        "n_speeches_total": int(len(speech_df)),
        "n_valid": int(valid.count()),
        "n_missing": int(values.isna().sum()),
        "n_score_1_or_2": int(score_1_or_2.sum()),
        "share_score_1_or_2": score_1_or_2.mean(),
        "percent_score_1_or_2": (
            score_1_or_2.mean() * 100
        ),
        "mean": valid.mean(),
        "sd": valid.std(ddof=1),
        "min": valid.min(),
        "median": valid.median(),
        "max": valid.max(),
    })

populism_prevalence = pd.DataFrame(
    populism_prevalence_rows
)

populism_prevalence = (
    populism_prevalence
    .sort_values(
        "mean",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# 14. GAL-TAN-THEMENHÄUFIGKEIT AUF REDEEBENE
#
# Verwendet die vorhandenen has_*-Variablen.
# ============================================================

gal_tan_topic_rows = []

for variable, indicator_variable in (
    GAL_TAN_TOPIC_INDICATORS.items()
):
    indicator = pd.to_numeric(
        speech_df[indicator_variable],
        errors="coerce",
    )

    score = pd.to_numeric(
        speech_df[variable],
        errors="coerce",
    )

    mentioned = indicator.eq(1)

    valid_scores = score.loc[
        mentioned & score.notna()
    ]

    gal_tan_topic_rows.append({
        "construct": "GAL-TAN",
        "variable": variable,
        "variable_label": VARIABLE_LABELS[variable],
        "indicator_variable": indicator_variable,
        "n_speeches_total": int(len(speech_df)),
        "n_topic_mentioned": int(mentioned.sum()),
        "share_topic_mentioned": mentioned.mean(),
        "percent_topic_mentioned": (
            mentioned.mean() * 100
        ),
        "n_valid_scores": int(valid_scores.count()),
        "mean": valid_scores.mean(),
        "sd": valid_scores.std(ddof=1),
        "min": valid_scores.min(),
        "median": valid_scores.median(),
        "max": valid_scores.max(),
    })

gal_tan_topic_prevalence = pd.DataFrame(
    gal_tan_topic_rows
)

gal_tan_topic_prevalence = (
    gal_tan_topic_prevalence
    .sort_values(
        "percent_topic_mentioned",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# 15. PARTEIRANGLISTEN AUF REDEEBENE
# ============================================================

speech_party_rankings = create_party_rankings(
    data=speech_df,
    variables=ALL_SCORE_VARIABLES,
)

populism_party_rankings = (
    speech_party_rankings.loc[
        speech_party_rankings["construct"]
        == "Populismus"
    ]
    .copy()
    .reset_index(drop=True)
)

gal_tan_party_rankings = (
    speech_party_rankings.loc[
        speech_party_rankings["construct"]
        == "GAL-TAN"
    ]
    .copy()
    .reset_index(drop=True)
)


# ============================================================
# 16. MAINSTREAM-WERTE AUF REDEEBENE
# ============================================================

mainstream_speech_stats = (
    create_mainstream_speech_stats(
        data=speech_df,
        variables=ALL_SCORE_VARIABLES,
    )
)

mainstream_party_balanced_stats = (
    create_party_balanced_mainstream_stats(
        data=speech_df,
        variables=ALL_SCORE_VARIABLES,
    )
)


# ============================================================
# 17. WORTANZAHL UND ANZAHL DER REDEN
# ============================================================

word_count_overall = (
    descriptive_stats_for_series(
        speech_df["word_count"]
    )
    .to_frame()
    .T
)

word_count_overall.insert(
    0,
    "variable",
    "word_count",
)

word_count_overall.insert(
    1,
    "variable_label",
    "Wortanzahl",
)

speech_counts_by_party = (
    speech_df
    .groupby(
        "party_clean",
        observed=False,
    )
    .size()
    .reset_index(
        name="n_speeches"
    )
)

speech_counts_by_party["share_speeches"] = (
    speech_counts_by_party["n_speeches"]
    / speech_counts_by_party["n_speeches"].sum()
)

speech_counts_by_party["percent_speeches"] = (
    speech_counts_by_party["share_speeches"]
    * 100
)

speech_counts_by_party = (
    speech_counts_by_party
    .sort_values("party_clean")
    .reset_index(drop=True)
)

speech_counts_by_wahlperiode = (
    speech_df
    .groupby(
        "wahlperiode",
        dropna=False,
    )
    .size()
    .reset_index(
        name="n_speeches"
    )
)

speech_counts_by_wahlperiode["share_speeches"] = (
    speech_counts_by_wahlperiode["n_speeches"]
    / speech_counts_by_wahlperiode["n_speeches"].sum()
)

speech_counts_by_wahlperiode["percent_speeches"] = (
    speech_counts_by_wahlperiode["share_speeches"]
    * 100
)

speech_counts_by_wahlperiode = (
    speech_counts_by_wahlperiode
    .sort_values("wahlperiode")
    .reset_index(drop=True)
)

speech_counts_by_party_wahlperiode = (
    speech_df
    .groupby(
        [
            "wahlperiode",
            "party_clean",
        ],
        observed=False,
        dropna=False,
    )
    .size()
    .reset_index(
        name="n_speeches"
    )
)

speech_counts_by_party_wahlperiode[
    "n_total_wahlperiode"
] = (
    speech_counts_by_party_wahlperiode
    .groupby("wahlperiode")["n_speeches"]
    .transform("sum")
)

speech_counts_by_party_wahlperiode[
    "share_within_wahlperiode"
] = np.where(
    speech_counts_by_party_wahlperiode[
        "n_total_wahlperiode"
    ] > 0,
    (
        speech_counts_by_party_wahlperiode[
            "n_speeches"
        ]
        / speech_counts_by_party_wahlperiode[
            "n_total_wahlperiode"
        ]
    ),
    np.nan,
)

speech_counts_by_party_wahlperiode[
    "percent_within_wahlperiode"
] = (
    speech_counts_by_party_wahlperiode[
        "share_within_wahlperiode"
    ]
    * 100
)

speech_counts_by_party_wahlperiode = (
    speech_counts_by_party_wahlperiode
    .sort_values(
        [
            "wahlperiode",
            "party_clean",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 18. PARTEI-INTERVALL-EBENE:
#     GESAMTSCORES UND DISTANZEN
# ============================================================

party_interval_overall_stats = (
    create_interval_overall_stats(
        data=party_interval_df,
        variable_mapping=INTERVAL_OVERALL_VARIABLES,
        analysis_level="party_interval",
    )
)

party_interval_overall_by_party = (
    create_interval_overall_by_party(
        data=party_interval_df,
        variable_mapping=INTERVAL_OVERALL_VARIABLES,
    )
)


# ============================================================
# 19. PARTEI-INTERVALL-EBENE:
#     SUBDIMENSIONEN UND DISTANZEN
# ============================================================

party_interval_subdimension_stats = (
    create_interval_subdimension_stats(
        data=party_interval_df,
        variable_mapping=(
            INTERVAL_SUBDIMENSION_VARIABLES
        ),
        analysis_level="party_interval",
    )
)

party_interval_subdimensions_by_party = (
    create_interval_subdimensions_by_party(
        data=party_interval_df,
        variable_mapping=(
            INTERVAL_SUBDIMENSION_VARIABLES
        ),
    )
)


# ============================================================
# 20. UNTERSTÜTZUNGSVERÄNDERUNG
# ============================================================

(
    support_change_descriptives,
    support_change_categories,
) = create_support_change_stats(
    data=party_interval_df
)


# ============================================================
# 21. AGGREGIERTE MAINSTREAM-EBENE:
#     GESAMTSCORES UND DISTANZEN
# ============================================================

mainstream_interval_overall_stats = (
    create_interval_overall_stats(
        data=mainstream_interval_df,
        variable_mapping=INTERVAL_OVERALL_VARIABLES,
        analysis_level=(
            "aggregated_mainstream_interval"
        ),
    )
)


# ============================================================
# 22. AGGREGIERTE MAINSTREAM-EBENE:
#     SUBDIMENSIONEN UND DISTANZEN
# ============================================================

mainstream_interval_subdimension_stats = (
    create_interval_subdimension_stats(
        data=mainstream_interval_df,
        variable_mapping=(
            INTERVAL_SUBDIMENSION_VARIABLES
        ),
        analysis_level=(
            "aggregated_mainstream_interval"
        ),
    )
)


# ============================================================
# 23. KONSTRUKTSPEZIFISCHE TABELLEN
# ============================================================

party_interval_populism = (
    party_interval_overall_stats.loc[
        party_interval_overall_stats["construct"]
        == "Populismus"
    ]
    .copy()
    .reset_index(drop=True)
)

party_interval_gal_tan = (
    party_interval_overall_stats.loc[
        party_interval_overall_stats["construct"]
        == "GAL-TAN"
    ]
    .copy()
    .reset_index(drop=True)
)

party_interval_pop_subdimensions = (
    party_interval_subdimension_stats.loc[
        party_interval_subdimension_stats["construct"]
        == "Populismus"
    ]
    .copy()
    .reset_index(drop=True)
)

party_interval_gal_subdimensions = (
    party_interval_subdimension_stats.loc[
        party_interval_subdimension_stats["construct"]
        == "GAL-TAN"
    ]
    .copy()
    .reset_index(drop=True)
)

mainstream_interval_populism = (
    mainstream_interval_overall_stats.loc[
        mainstream_interval_overall_stats["construct"]
        == "Populismus"
    ]
    .copy()
    .reset_index(drop=True)
)

mainstream_interval_gal_tan = (
    mainstream_interval_overall_stats.loc[
        mainstream_interval_overall_stats["construct"]
        == "GAL-TAN"
    ]
    .copy()
    .reset_index(drop=True)
)

mainstream_interval_pop_subdimensions = (
    mainstream_interval_subdimension_stats.loc[
        mainstream_interval_subdimension_stats[
            "construct"
        ]
        == "Populismus"
    ]
    .copy()
    .reset_index(drop=True)
)

mainstream_interval_gal_subdimensions = (
    mainstream_interval_subdimension_stats.loc[
        mainstream_interval_subdimension_stats[
            "construct"
        ]
        == "GAL-TAN"
    ]
    .copy()
    .reset_index(drop=True)
)


# ============================================================
# 24. CSV-DATEIEN SPEICHERN
# ============================================================

# ------------------------------------------------------------
# Redeebene: allgemeine Statistiken
# ------------------------------------------------------------

write_csv(
    speech_scores_overall,
    "01_scores_overall",
)

write_csv(
    speech_scores_by_party,
    "02_scores_by_party",
)


# ------------------------------------------------------------
# Redeebene: Populismus-Prävalenz und GAL-TAN-Themen
# ------------------------------------------------------------

write_csv(
    populism_prevalence,
    "03_populism_prevalence",
)

write_csv(
    gal_tan_topic_prevalence,
    "04_gal_tan_topics",
)


# ------------------------------------------------------------
# Redeebene: Parteiranglisten
# ------------------------------------------------------------

write_csv(
    populism_party_rankings,
    "05_populism_party_rankings",
)

write_csv(
    gal_tan_party_rankings,
    "06_gal_tan_party_rankings",
)

write_csv(
    speech_party_rankings,
    "07_all_party_rankings",
)


# ------------------------------------------------------------
# Redeebene: Mainstream-Parteien
# ------------------------------------------------------------

write_csv(
    mainstream_speech_stats,
    "08_mainstream_speech_weighted",
)

write_csv(
    mainstream_party_balanced_stats,
    "09_mainstream_party_weighted",
)


# ------------------------------------------------------------
# Wortanzahl und Anzahl der Reden
# ------------------------------------------------------------

write_csv(
    word_count_overall,
    "10_word_count_overall",
)

write_csv(
    speech_counts_by_party,
    "11_speech_counts_by_party",
)

write_csv(
    speech_counts_by_wahlperiode,
    "12_speech_counts_by_wahlperiode",
)

write_csv(
    speech_counts_by_party_wahlperiode,
    "13_speech_counts_by_party_wahlperiode",
)


# ------------------------------------------------------------
# Partei-Intervall-Ebene: Gesamtscores
# ------------------------------------------------------------

write_csv(
    party_interval_populism,
    "14_party_interval_populism",
)

write_csv(
    party_interval_gal_tan,
    "15_party_interval_gal_tan",
)

write_csv(
    party_interval_overall_stats,
    "16_party_interval_overall",
)

write_csv(
    party_interval_overall_by_party,
    "17_party_interval_by_party",
)


# ------------------------------------------------------------
# Partei-Intervall-Ebene: Subdimensionen
# ------------------------------------------------------------

write_csv(
    party_interval_pop_subdimensions,
    "18_party_interval_populism_subdimensions",
)

write_csv(
    party_interval_gal_subdimensions,
    "19_party_interval_gal_tan_subdimensions",
)

write_csv(
    party_interval_subdimension_stats,
    "20_party_interval_all_subdimensions",
)

write_csv(
    party_interval_subdimensions_by_party,
    "21_party_interval_subdimensions_by_party",
)


# ------------------------------------------------------------
# Unterstützungsveränderung
# ------------------------------------------------------------

write_csv(
    support_change_descriptives,
    "22_support_change_descriptives",
)

write_csv(
    support_change_categories,
    "23_support_change_categories",
)


# ------------------------------------------------------------
# Aggregierte Mainstream-Ebene: Gesamtscores
# ------------------------------------------------------------

write_csv(
    mainstream_interval_populism,
    "24_mainstream_interval_populism",
)

write_csv(
    mainstream_interval_gal_tan,
    "25_mainstream_interval_gal_tan",
)

write_csv(
    mainstream_interval_overall_stats,
    "26_mainstream_interval_overall",
)


# ------------------------------------------------------------
# Aggregierte Mainstream-Ebene: Subdimensionen
# ------------------------------------------------------------

write_csv(
    mainstream_interval_pop_subdimensions,
    "27_mainstream_interval_populism_subdimensions",
)

write_csv(
    mainstream_interval_gal_subdimensions,
    "28_mainstream_interval_gal_tan_subdimensions",
)

write_csv(
    mainstream_interval_subdimension_stats,
    "29_mainstream_interval_all_subdimensions",
)


# ============================================================
# 25. KONTROLLAUSGABEN
# ============================================================

print("\n" + "=" * 78)
print("ERGEBNISRELEVANTE DESKRIPTIVE STATISTIKEN")
print("=" * 78)


print("\n1. Populismus insgesamt auf Redeebene:")

print(
    speech_scores_overall.loc[
        speech_scores_overall["construct"]
        == "Populismus",
        [
            "variable_label",
            "n_valid",
            "mean",
            "sd",
            "min",
            "median",
            "max",
        ],
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n2. Populismus-Prävalenz:")

print(
    populism_prevalence[
        [
            "variable_label",
            "n_valid",
            "n_score_1_or_2",
            "percent_score_1_or_2",
            "mean",
            "sd",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n3. GAL-TAN insgesamt auf Redeebene:")

print(
    speech_scores_overall.loc[
        speech_scores_overall["construct"]
        == "GAL-TAN",
        [
            "variable_label",
            "n_valid",
            "mean",
            "sd",
            "min",
            "median",
            "max",
        ],
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n4. GAL-TAN-Themenhäufigkeiten:")

print(
    gal_tan_topic_prevalence[
        [
            "variable_label",
            "n_topic_mentioned",
            "percent_topic_mentioned",
            "mean",
            "sd",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n5. Populismus nach Partei:")

print(
    populism_party_rankings[
        [
            "variable_label",
            "party_clean",
            "n_valid",
            "mean",
            "sd",
            "rank_highest_value",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n6. GAL-TAN nach Partei:")

print(
    gal_tan_party_rankings[
        [
            "variable_label",
            "party_clean",
            "n_valid",
            "mean",
            "sd",
            "rank_highest_value",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n7. Mainstream-Werte auf Redeebene:")

print(
    mainstream_speech_stats[
        [
            "construct",
            "variable_label",
            "n_valid",
            "mean",
            "sd",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print(
    "\n8. Partei-Intervall-Ebene: "
    "Gesamtscores und Distanzen"
)

print(
    party_interval_overall_stats[
        [
            "construct",
            "measure_type",
            "variable",
            "n_valid",
            "mean",
            "sd",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print(
    "\n9. Partei-Intervall-Ebene: "
    "Subdimensionswerte und Distanzen"
)

print(
    party_interval_subdimension_stats[
        [
            "construct",
            "subdimension_label",
            "measure_type",
            "n_valid",
            "mean",
            "sd",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n10. Unterstützungsveränderung:")

print(
    support_change_descriptives
    .round(DECIMALS)
    .to_string(index=False)
)


print(
    "\n11. Verluste, unveränderte Werte "
    "und Gewinne:"
)

print(
    support_change_categories
    .round(DECIMALS)
    .to_string(index=False)
)


print(
    "\n12. Aggregierte Mainstream-Ebene: "
    "Gesamtscores und Distanzen"
)

print(
    mainstream_interval_overall_stats[
        [
            "construct",
            "measure_type",
            "variable",
            "n_valid",
            "mean",
            "sd",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print(
    "\n13. Aggregierte Mainstream-Ebene: "
    "Subdimensionswerte und Distanzen"
)

print(
    mainstream_interval_subdimension_stats[
        [
            "construct",
            "subdimension_label",
            "measure_type",
            "n_valid",
            "mean",
            "sd",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n14. Anzahl der Reden pro Partei:")

print(
    speech_counts_by_party
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n15. Anzahl der Reden pro Wahlperiode:")

print(
    speech_counts_by_wahlperiode
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n16. Wortanzahl:")

print(
    word_count_overall
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n" + "=" * 78)
print("AUSWERTUNG ABGESCHLOSSEN")
print("=" * 78)

print(
    f"\nCSV-Dateien gespeichert unter:\n"
    f"{OUTPUT_DIR.resolve()}"
)