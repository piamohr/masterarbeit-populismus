# ============================================================
# H1 AUF AGGREGIERTER MAINSTREAM-EBENE
#
# H1:
# Je größer die Unterstützung der AfD ist, desto stärker nähern
# sich die Mainstream-Parteien der AfD positionell an.
#
# Die Hypothese wird mit zwei komplementären
# Operationalisierungen geprüft:
#
# MODELL A: Positionelle Distanz zur AfD
#   AV: gal_tan_distance_to_afd
#   Erwartung: Koeffizient von afd_support_t < 0
#
# MODELL B: Absolute GAL-TAN-Position der Mainstream-Parteien
#   AV: mean_gal_tan_score
#   Erwartung: Koeffizient von afd_support_t > 0
#
# Voraussetzung für Modell B:
#   Höhere Werte der GAL-TAN-Skala stehen für stärker
#   traditionalistische, autoritäre und nationalistische Positionen.
#
# Analyseeinheit:
# Umfrageintervall, aggregiert über alle Mainstream-Parteien
#
# Inferenz:
# Newey-West-/HAC-Standardfehler, da nur eine aggregierte
# Zeitreihe und keine Parteicluster vorliegen.
#
# Für beide Operationalisierungen werden geschätzt:
#   1. Hauptmodell mit Wahlperioden-Fixed-Effects
#   2. HAC-Sensitivitätsanalyse mit mehreren Lag-Längen
#
# Alle Ergebniswerte werden mit drei Nachkommastellen
# ausgegeben und gespeichert.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ============================================================
# 1. EINSTELLUNGEN
# ============================================================

INPUT_FILE = Path(
    "data/processed/final_analysis/"
    "06_mainstream_polling_interval_distances.csv"
)

OUTPUT_DIR = Path("results/hypothesis1_mainstream")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTOR = "afd_support_t"
PERIOD = "wahlperiode_cat"
DATE = "poll_date"

ALPHA = 0.05
DECIMALS = 3

MODEL_SPECS = {
    "distance_to_afd": {
        "label": "Positionelle Distanz zur AfD",
        "outcome": "gal_tan_distance_to_afd",
        "expected_direction": "negative",
        "expected_sign": -1,
        "output_prefix": "h1_mainstream_distance",
        "descriptive_variables": [
            "gal_tan_distance_to_afd",
            "afd_support_t",
            "mean_gal_tan_score",
            "afd_mean_gal_tan_score",
        ],
    },
    "absolute_gal_tan": {
        "label": "Absolute GAL-TAN-Position",
        "outcome": "mean_gal_tan_score",
        "expected_direction": "positive",
        "expected_sign": 1,
        "output_prefix": "h1_mainstream_absolute_gal_tan",
        "descriptive_variables": [
            "mean_gal_tan_score",
            "afd_support_t",
            "afd_mean_gal_tan_score",
            "gal_tan_distance_to_afd",
        ],
    },
}


# ============================================================
# 2. HILFSFUNKTIONEN
# ============================================================

def automatic_hac_lag(n_observations: int) -> int:
    """
    Automatische Newey-West-Lag-Auswahl nach der Faustregel:

    floor(4 * (n / 100)^(2/9))

    Mindestens ein Lag wird verwendet.
    """
    lag = int(
        np.floor(
            4 * (n_observations / 100) ** (2 / 9)
        )
    )

    return max(1, lag)


def extract_parameter(
    result,
    parameter_name: str,
) -> dict:
    """
    Extrahiert Koeffizient, Standardfehler, Teststatistik,
    p-Wert und Konfidenzintervall anhand des Parameternamens.
    """
    if parameter_name not in result.params.index:
        raise ValueError(
            f"Parameter '{parameter_name}' wurde im Modell "
            "nicht gefunden.\n"
            f"Vorhandene Parameter: {list(result.params.index)}"
        )

    confidence_interval = result.conf_int(
        alpha=ALPHA
    ).loc[parameter_name]

    return {
        "coefficient": float(
            result.params.loc[parameter_name]
        ),
        "standard_error": float(
            result.bse.loc[parameter_name]
        ),
        "test_statistic": float(
            result.tvalues.loc[parameter_name]
        ),
        "p_value": float(
            result.pvalues.loc[parameter_name]
        ),
        "ci_95_low": float(
            confidence_interval.iloc[0]
        ),
        "ci_95_high": float(
            confidence_interval.iloc[1]
        ),
    }


def fit_hac_model(
    formula: str,
    data: pd.DataFrame,
    maxlags: int,
):
    """
    Schätzt ein OLS-Modell mit HAC-/Newey-West-
    Standardfehlern.
    """
    model = smf.ols(
        formula=formula,
        data=data,
    )

    return model.fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": maxlags,
            "use_correction": True,
        },
        use_t=True,
    )


def round_numeric_columns(
    dataframe: pd.DataFrame,
    decimals: int = DECIMALS,
) -> pd.DataFrame:
    """
    Rundet alle numerischen Spalten eines DataFrames.
    """
    rounded = dataframe.copy()

    numeric_columns = rounded.select_dtypes(
        include=[np.number]
    ).columns

    rounded[numeric_columns] = rounded[
        numeric_columns
    ].round(decimals)

    return rounded


def save_result_dataframe(
    dataframe: pd.DataFrame,
    output_path: Path,
):
    """
    Speichert numerische Ergebniswerte mit drei
    Nachkommastellen als CSV.
    """
    round_numeric_columns(
        dataframe,
        decimals=DECIMALS,
    ).to_csv(
        output_path,
        index=False,
        float_format="%.3f",
    )


def direction_matches(
    coefficient: float,
    expected_sign: int,
) -> bool:
    """
    Prüft, ob der geschätzte Koeffizient in die
    erwartete Richtung weist.
    """
    if expected_sign == 1:
        return coefficient > 0

    if expected_sign == -1:
        return coefficient < 0

    raise ValueError(
        "expected_sign muss entweder 1 oder -1 sein."
    )


def interpret_result(
    coefficient: float,
    p_value: float,
    expected_sign: int,
    label: str,
) -> str:
    """
    Erstellt eine richtungsbezogene Ergebnisinterpretation.
    """
    matches = direction_matches(
        coefficient=coefficient,
        expected_sign=expected_sign,
    )

    significant = p_value < ALPHA

    if matches and significant:
        return (
            f"{label}: Der Koeffizient weist in die erwartete "
            "Richtung und ist statistisch signifikant."
        )

    if matches and not significant:
        return (
            f"{label}: Der Koeffizient weist in die erwartete "
            "Richtung, ist aber nicht statistisch signifikant."
        )

    if not matches and significant:
        return (
            f"{label}: Der Koeffizient ist statistisch signifikant, "
            "weist aber in die der Hypothese entgegengesetzte Richtung."
        )

    return (
        f"{label}: Der Koeffizient weist nicht in die erwartete "
        "Richtung und ist nicht statistisch signifikant."
    )


def prepare_analysis_sample(
    data: pd.DataFrame,
    outcome: str,
    descriptive_variables: list[str],
) -> pd.DataFrame:
    """
    Erstellt die vollständige und chronologisch sortierte
    Analysestichprobe für eine abhängige Variable.

    Neben den Modellvariablen werden vorhandene deskriptive
    Variablen in der gespeicherten Stichprobe beibehalten.
    """
    required_columns = [
        outcome,
        PREDICTOR,
        PERIOD,
        DATE,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Folgende Variablen fehlen:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_columns
            )
        )

    retained_columns = list(
        dict.fromkeys(
            required_columns + descriptive_variables
        )
    )

    missing_retained_columns = [
        column
        for column in retained_columns
        if column not in data.columns
    ]

    if missing_retained_columns:
        raise ValueError(
            "Folgende Variablen fehlen:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_retained_columns
            )
        )

    analysis_df = data[
        retained_columns
    ].copy()

    analysis_df[DATE] = pd.to_datetime(
        analysis_df[DATE],
        errors="coerce",
    )

    analysis_df[outcome] = pd.to_numeric(
        analysis_df[outcome],
        errors="coerce",
    )

    analysis_df[PREDICTOR] = pd.to_numeric(
        analysis_df[PREDICTOR],
        errors="coerce",
    )

    analysis_df = (
        analysis_df
        .dropna(
            subset=[
                outcome,
                PREDICTOR,
                PERIOD,
                DATE,
            ]
        )
        .sort_values(DATE)
        .reset_index(drop=True)
    )

    analysis_df[PERIOD] = (
        analysis_df[PERIOD]
        .astype(str)
        .astype("category")
    )

    if len(analysis_df) < 10:
        raise ValueError(
            f"Für die Zeitreihenanalyse von '{outcome}' liegen "
            "weniger als zehn vollständige Beobachtungen vor."
        )

    return analysis_df


def print_model_result(
    heading: str,
    stats: dict,
    interpretation: str,
):
    """
    Gibt ein Modellergebnis einheitlich in der Konsole aus.
    """
    print("\n" + "=" * 72)
    print(heading)
    print("=" * 72)

    print(
        f"Koeffizient:        "
        f"{stats['coefficient']:.3f}"
    )
    print(
        f"HAC-Standardfehler: "
        f"{stats['standard_error']:.3f}"
    )
    print(
        f"t-Wert:             "
        f"{stats['test_statistic']:.3f}"
    )
    print(
        f"p-Wert:             "
        f"{stats['p_value']:.3f}"
    )
    print(
        "95%-KI:             "
        f"[{stats['ci_95_low']:.3f}, "
        f"{stats['ci_95_high']:.3f}]"
    )
    print("\nInterpretation:")
    print(interpretation)


def run_h1_model(
    data: pd.DataFrame,
    model_key: str,
    model_spec: dict,
) -> dict:
    """
    Führt Hauptmodell und HAC-Sensitivitätsanalyse für eine
    H1-Operationalisierung aus.
    """
    label = model_spec["label"]
    outcome = model_spec["outcome"]
    expected_direction = model_spec["expected_direction"]
    expected_sign = model_spec["expected_sign"]
    prefix = model_spec["output_prefix"]
    descriptive_variables = model_spec[
        "descriptive_variables"
    ]

    output_main_model = (
        OUTPUT_DIR / f"{prefix}_main_model.csv"
    )
    output_hac_sensitivity = (
        OUTPUT_DIR / f"{prefix}_hac_sensitivity.csv"
    )
    output_analysis_sample = (
        OUTPUT_DIR / f"{prefix}_analysis_sample.csv"
    )

    analysis_df = prepare_analysis_sample(
        data=data,
        outcome=outcome,
        descriptive_variables=descriptive_variables,
    )

    n_observations = len(analysis_df)
    main_hac_lag = automatic_hac_lag(n_observations)

    print("\n\n" + "#" * 72)
    print(f"STARTE H1-MODELL: {label.upper()}")
    print("#" * 72)

    print("\n" + "=" * 72)
    print(f"H1: Aggregierte Mainstream-Ebene – {label}")
    print("=" * 72)

    print(f"Abhängige Variable: {outcome}")
    print(f"Analyseeinheiten: {n_observations}")
    print(
        f"Zeitraum: "
        f"{analysis_df[DATE].min().date()} bis "
        f"{analysis_df[DATE].max().date()}"
    )
    print(
        f"Wahlperioden: "
        f"{analysis_df[PERIOD].nunique()}"
    )
    print(
        f"Automatisch gewählter HAC-Lag: "
        f"{main_hac_lag}"
    )
    print(
        f"Erwartete Richtung: "
        f"{expected_direction}"
    )

    print("\nDeskriptive Werte:")
    print(
        analysis_df[descriptive_variables]
        .describe()
        .round(DECIMALS)
    )

    # --------------------------------------------------------
    # Hauptmodell mit Wahlperioden-Fixed-Effects
    # --------------------------------------------------------

    main_formula = f"""
        {outcome}
        ~ {PREDICTOR}
        + C({PERIOD})
    """

    main_result = fit_hac_model(
        formula=main_formula,
        data=analysis_df,
        maxlags=main_hac_lag,
    )

    main_stats = extract_parameter(
        main_result,
        PREDICTOR,
    )

    main_supported = (
        direction_matches(
            coefficient=main_stats["coefficient"],
            expected_sign=expected_sign,
        )
        and main_stats["p_value"] < ALPHA
    )

    main_interpretation = interpret_result(
        coefficient=main_stats["coefficient"],
        p_value=main_stats["p_value"],
        expected_sign=expected_sign,
        label=label,
    )

    print("\nModellspezifikation:")
    print(main_formula.strip())

    print_model_result(
        heading=(
            f"H1: Hauptmodell mit Wahlperioden-Fixed-Effects "
            f"– {label}"
        ),
        stats=main_stats,
        interpretation=main_interpretation,
    )

    # --------------------------------------------------------
    # HAC-Sensitivitätsanalyse
    # --------------------------------------------------------

    candidate_lags = sorted(
        {
            1,
            2,
            3,
            main_hac_lag,
            6,
            12,
        }
    )

    candidate_lags = [
        lag
        for lag in candidate_lags
        if lag < n_observations
    ]

    sensitivity_rows = []

    for lag in candidate_lags:
        sensitivity_result = fit_hac_model(
            formula=main_formula,
            data=analysis_df,
            maxlags=lag,
        )

        sensitivity_stats = extract_parameter(
            sensitivity_result,
            PREDICTOR,
        )

        expected_direction_result = direction_matches(
            coefficient=sensitivity_stats["coefficient"],
            expected_sign=expected_sign,
        )

        sensitivity_rows.append(
            {
                "hypothesis": "H1",
                "operationalization": model_key,
                "model_label": label,
                "outcome": outcome,
                "expected_direction": expected_direction,
                "hac_lag": lag,
                **sensitivity_stats,
                "direction_matches_expectation": (
                    expected_direction_result
                ),
                "significant_05": (
                    sensitivity_stats["p_value"] < ALPHA
                ),
                "supported": (
                    expected_direction_result
                    and sensitivity_stats["p_value"] < ALPHA
                ),
            }
        )

    hac_sensitivity = pd.DataFrame(
        sensitivity_rows
    )

    print("\n" + "=" * 72)
    print(f"HAC-Sensitivitätsanalyse – {label}")
    print("=" * 72)

    print(
        hac_sensitivity[
            [
                "hac_lag",
                "coefficient",
                "standard_error",
                "test_statistic",
                "p_value",
                "ci_95_low",
                "ci_95_high",
                "direction_matches_expectation",
                "significant_05",
                "supported",
            ]
        ]
        .round(DECIMALS)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Ergebnisse speichern
    # --------------------------------------------------------

    main_output = pd.DataFrame(
        [
            {
                "hypothesis": "H1",
                "level": "mainstream_polling_interval",
                "operationalization": model_key,
                "model_label": label,
                "model": "wahlperiode_fixed_effects",
                "outcome": outcome,
                "predictor": PREDICTOR,
                "expected_direction": expected_direction,
                "hac_lag": main_hac_lag,
                **main_stats,
                "n_observations": n_observations,
                "n_periods": (
                    analysis_df[PERIOD].nunique()
                ),
                "r_squared": float(
                    main_result.rsquared
                ),
                "adjusted_r_squared": float(
                    main_result.rsquared_adj
                ),
                "supported": main_supported,
                "interpretation": main_interpretation,
            }
        ]
    )

    save_result_dataframe(
        main_output,
        output_main_model,
    )

    save_result_dataframe(
        hac_sensitivity,
        output_hac_sensitivity,
    )

    analysis_df.to_csv(
        output_analysis_sample,
        index=False,
        float_format="%.3f",
    )

    print("\nGespeicherte Dateien:")
    print(f"- {output_main_model}")
    print(f"- {output_hac_sensitivity}")
    print(f"- {output_analysis_sample}")

    return {
        "main": main_output,
        "sensitivity": hac_sensitivity,
    }


# ============================================================
# 3. DATEN EINLESEN
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Datei nicht gefunden:\n{INPUT_FILE.resolve()}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 72)
print("H1: Aggregierte Mainstream-Analyse")
print("=" * 72)
print(f"Datei: {INPUT_FILE}")
print(f"Zeilen im Gesamtdatensatz: {len(df)}")
print(f"Spalten im Gesamtdatensatz: {len(df.columns)}")


# ============================================================
# 4. BEIDE H1-OPERATIONALISIERUNGEN SCHÄTZEN
# ============================================================

all_main_results = []
all_sensitivity_results = []

for model_key, model_spec in MODEL_SPECS.items():
    results = run_h1_model(
        data=df,
        model_key=model_key,
        model_spec=model_spec,
    )

    all_main_results.append(
        results["main"]
    )
    all_sensitivity_results.append(
        results["sensitivity"]
    )


# ============================================================
# 5. GEMEINSAME ERGEBNISTABELLEN
# ============================================================

combined_main_results = pd.concat(
    all_main_results,
    ignore_index=True,
)

combined_sensitivity_results = pd.concat(
    all_sensitivity_results,
    ignore_index=True,
)

OUTPUT_COMBINED_MAIN = (
    OUTPUT_DIR / "h1_mainstream_combined_main_models.csv"
)

OUTPUT_COMBINED_SENSITIVITY = (
    OUTPUT_DIR / "h1_mainstream_combined_hac_sensitivity.csv"
)

save_result_dataframe(
    combined_main_results,
    OUTPUT_COMBINED_MAIN,
)

save_result_dataframe(
    combined_sensitivity_results,
    OUTPUT_COMBINED_SENSITIVITY,
)


# ============================================================
# 6. ABSCHLUSSAUSGABE
# ============================================================

print("\n\n" + "=" * 72)
print("H1: ALLE AGGREGIERTEN MAINSTREAM-ANALYSEN ABGESCHLOSSEN")
print("=" * 72)

print("\nHauptmodelle im Vergleich:")
print(
    combined_main_results[
        [
            "model_label",
            "outcome",
            "expected_direction",
            "coefficient",
            "standard_error",
            "test_statistic",
            "p_value",
            "ci_95_low",
            "ci_95_high",
            "n_observations",
            "hac_lag",
            "supported",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)

print("\nGemeinsame Ergebnisdateien:")
print(f"- Hauptmodelle:       {OUTPUT_COMBINED_MAIN}")
print(f"- HAC-Sensitivität:   {OUTPUT_COMBINED_SENSITIVITY}")

print(
    "\nHinweis zur Interpretation:\n"
    "- Beim Distanzmodell unterstützt ein negativer "
    "Koeffizient H1.\n"
    "- Beim Modell der absoluten GAL-TAN-Position unterstützt "
    "ein positiver Koeffizient H1, sofern höhere Werte den "
    "TAN-Pol abbilden.\n"
    "- Beide Operationalisierungen beantworten unterschiedliche, "
    "aber komplementäre Fragen."
)