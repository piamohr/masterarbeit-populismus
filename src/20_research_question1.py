# ============================================================
# FF1: GAL-TAN-SUBDIMENSIONEN
# AUF PARTEI-INTERVALL-EBENE
#
# Forschungsfrage:
# Unterscheiden sich die GAL-TAN-Subdimensionen hinsichtlich
# ihres Zusammenhangs mit der Unterstützung der AfD?
#
# Die Forschungsfrage wird mit zwei komplementären
# Operationalisierungen untersucht:
#
# MODELLGRUPPE A: Distanz zur AfD
#   AVs: dimensionsspezifische Distanzen zur AfD
#   Interpretation:
#   Ein negativer Koeffizient bedeutet, dass die Distanz zur AfD
#   mit steigender AfD-Unterstützung abnimmt.
#
# MODELLGRUPPE B: Absolute GAL-TAN-Positionen
#   AVs: absolute Mittelwerte der fünf GAL-TAN-Subdimensionen
#   Interpretation:
#   Ein positiver Koeffizient bedeutet, dass sich die jeweilige
#   Position mit steigender AfD-Unterstützung in Richtung TAN
#   verschiebt, sofern höhere Werte den TAN-Pol abbilden.
#
# Vorgehen für beide Operationalisierungen:
# 1. Separates Fixed-Effects-Modell je Subdimension
# 2. Paarweise Differenzmodelle zum Vergleich der Koeffizienten
# 3. Partei-Fixed-Effects
# 4. Wahlperioden-Fixed-Effects
# 5. Clusterrobuste Standardfehler auf Parteiebene
# 6. Wild-Cluster-Bootstrap WCR11 und WCR31
# 7. Holm-Korrektur der dimensionsspezifischen und paarweisen Tests
#
# Alle Ergebniswerte werden mit drei Nachkommastellen
# ausgegeben und gespeichert.
# ============================================================

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from wildboottest.wildboottest import wildboottest


# ============================================================
# 1. EINSTELLUNGEN
# ============================================================

INPUT_FILE = Path(
    "data/processed/final_analysis/"
    "05_party_polling_interval_distances.csv"
)

OUTPUT_DIR = Path(
    "results/research_question1"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AFD_SUPPORT = "afd_support_t"
PARTY = "party_clean"
PERIOD = "wahlperiode_cat"
MAINSTREAM = "is_mainstream_party"

CENTERED_SUPPORT = "afd_support_c"

BOOTSTRAP_REPLICATIONS = 9999
SEED = 12345
ALPHA = 0.05
DECIMALS = 3

# True:
# Innerhalb jeder Operationalisierung verwenden alle fünf
# Dimensionsmodelle dieselben vollständigen Partei-Intervalle.
#
# False:
# Jede Dimension verwendet ihre maximal verfügbare Stichprobe.
USE_COMMON_COMPLETE_CASE_SAMPLE = True

MAINSTREAM_PARTIES = {
    "cdu/csu",
    "spd",
    "fdp",
    "grüne",
    "linke",
}

DIMENSION_LABELS = {
    "G1_environment": "Umwelt",
    "G2_migration": "Migration",
    "G3_society": "Gesellschaftspolitik",
    "G4_civil_rights": "Bürgerrechte",
    "G5_cosmopolitanism": "Kosmopolitismus",
}

# Distanzvariablen sind im bisherigen Datensatz eindeutig benannt.
DISTANCE_DIMENSIONS = {
    "G1_environment": "G1_environment_distance_to_afd",
    "G2_migration": "G2_migration_distance_to_afd",
    "G3_society": "G3_society_distance_to_afd",
    "G4_civil_rights": "G4_civil_rights_distance_to_afd",
    "G5_cosmopolitanism": (
        "G5_cosmopolitanism_distance_to_afd"
    ),
}

# Absolute Subdimensionswerte aus
# 05_party_polling_interval_distances.csv.
ABSOLUTE_DIMENSIONS = {
    "G1_environment": "mean_G1_environment",
    "G2_migration": "mean_G2_migration",
    "G3_society": "mean_G3_society",
    "G4_civil_rights": "mean_G4_civil_rights",
    "G5_cosmopolitanism": "mean_G5_cosmopolitanism",
}


# ============================================================
# 2. HILFSFUNKTIONEN
# ============================================================

def mainstream_mask(series: pd.Series) -> pd.Series:
    """Erzeugt einen robusten Filter für die Mainstream-Variable."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().any():
        return numeric.eq(1)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "ja"])
    )



def save_result_dataframe(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    """Speichert numerische Werte mit drei Nachkommastellen."""
    dataframe.to_csv(
        output_file,
        index=False,
        float_format="%.3f",
    )


def create_pairwise_report_table(
    pairwise_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Erstellt eine kompakte Berichtstabelle mit allen paarweisen
    Vergleichen der GAL-TAN-Subdimensionen.

    Die Tabelle enthält sowohl die clusterrobuste Standardinferenz
    als auch die WCR11- und WCR31-Ergebnisse einschließlich der
    jeweiligen Holm-Korrekturen.
    """
    report_table = pairwise_results.copy()

    operationalization_labels = {
        "distance": "Distanz zur AfD",
        "absolute": "Absolute GAL-TAN-Position",
    }

    report_table["Operationalisierung"] = (
        report_table["operationalization"]
        .map(operationalization_labels)
        .fillna(report_table["operationalization"])
    )

    report_table["Vergleich"] = (
        report_table["dimension_a_label"]
        + " – "
        + report_table["dimension_b_label"]
    )

    report_table["b_Differenz"] = (
        report_table["slope_difference"]
    )

    report_table["SE_cluster"] = (
        report_table["cluster_robust_se"]
    )

    report_table["KI_95_unten"] = (
        report_table["ci_95_low"]
    )

    report_table["KI_95_oben"] = (
        report_table["ci_95_high"]
    )

    report_table["p_cluster"] = (
        report_table["p_value"]
    )

    report_table["p_cluster_Holm"] = (
        report_table["p_value_holm"]
    )

    report_table["p_WCR11"] = (
        report_table["wcr11_p_value"]
    )

    report_table["p_WCR11_Holm"] = (
        report_table["wcr11_p_value_holm"]
    )

    report_table["p_WCR31"] = (
        report_table["wcr31_p_value"]
    )

    report_table["p_WCR31_Holm"] = (
        report_table["wcr31_p_value_holm"]
    )

    report_table["Signifikanz_cluster"] = np.where(
        report_table["p_value_holm"] < ALPHA,
        "*",
        "",
    )

    report_table["Signifikanz_WCR11"] = np.where(
        report_table["wcr11_p_value_holm"] < ALPHA,
        "*",
        "",
    )

    report_table["Signifikanz_WCR31"] = np.where(
        report_table["wcr31_p_value_holm"] < ALPHA,
        "†",
        "",
    )

    report_table = report_table[
        [
            "Operationalisierung",
            "Vergleich",
            "b_Differenz",
            "SE_cluster",
            "KI_95_unten",
            "KI_95_oben",
            "p_cluster",
            "p_cluster_Holm",
            "p_WCR11",
            "p_WCR11_Holm",
            "p_WCR31",
            "p_WCR31_Holm",
            "Signifikanz_cluster",
            "Signifikanz_WCR11",
            "Signifikanz_WCR31",
            "n_observations",
            "n_clusters",
            "n_periods",
        ]
    ].rename(
        columns={
            "n_observations": "N",
            "n_clusters": "Cluster",
            "n_periods": "Wahlperioden",
        }
    )

    operationalization_order = pd.CategoricalDtype(
        categories=[
            "Distanz zur AfD",
            "Absolute GAL-TAN-Position",
        ],
        ordered=True,
    )

    report_table["Operationalisierung"] = (
        report_table["Operationalisierung"]
        .astype(operationalization_order)
    )

    report_table = (
        report_table
        .sort_values(
            [
                "Operationalisierung",
                "Vergleich",
            ]
        )
        .reset_index(drop=True)
    )

    return report_table


def extract_parameter(
    ols_result,
    robust_result,
    parameter_name: str,
) -> dict:
    """Extrahiert die Kennwerte des interessierenden Parameters."""
    parameter_names = ols_result.model.exog_names

    if parameter_name not in parameter_names:
        raise ValueError(
            f"Parameter '{parameter_name}' wurde nicht gefunden."
        )

    parameter_index = parameter_names.index(
        parameter_name
    )

    confidence_interval = robust_result.conf_int(
        alpha=ALPHA
    )[parameter_index]

    return {
        "coefficient": float(
            robust_result.params[parameter_index]
        ),
        "cluster_robust_se": float(
            robust_result.bse[parameter_index]
        ),
        "t_value": float(
            robust_result.tvalues[parameter_index]
        ),
        "p_value": float(
            robust_result.pvalues[parameter_index]
        ),
        "ci_95_low": float(
            confidence_interval[0]
        ),
        "ci_95_high": float(
            confidence_interval[1]
        ),
    }


def fit_cluster_model(
    formula: str,
    data: pd.DataFrame,
):
    """
    Schätzt ein OLS-Modell mit Partei- und Wahlperioden-FE
    sowie clusterrobusten Standardfehlern.
    """
    model = smf.ols(
        formula=formula,
        data=data,
    )

    ols_result = model.fit()

    cluster_array = (
        data["cluster_id"]
        .to_numpy(dtype=np.int64)
    )

    if len(cluster_array) != int(ols_result.nobs):
        raise ValueError(
            "Die Zahl der Clusterkennungen stimmt nicht mit "
            "der Zahl der Modellbeobachtungen überein."
        )

    robust_result = (
        ols_result.get_robustcov_results(
            cov_type="cluster",
            groups=cluster_array,
            use_correction=True,
            df_correction=True,
            use_t=True,
        )
    )

    return (
        model,
        ols_result,
        robust_result,
        cluster_array,
    )


def extract_bootstrap_p_value(
    bootstrap_result,
    parameter_name: str,
) -> float:
    """Extrahiert den p-Wert aus wildboottest."""
    if not isinstance(
        bootstrap_result,
        pd.DataFrame,
    ):
        return np.nan

    if parameter_name in bootstrap_result.index:
        row = bootstrap_result.loc[
            parameter_name
        ]
    else:
        row = bootstrap_result.iloc[0]

    for column in [
        "p-value",
        "p_value",
        "pvalue",
    ]:
        if column in bootstrap_result.columns:
            return float(row[column])

    return np.nan


def run_wild_cluster_bootstrap(
    model,
    cluster_array: np.ndarray,
    parameter_name: str,
    bootstrap_type: str,
) -> float:
    """
    Führt einen Wild-Cluster-Bootstrap aus und gibt den
    extrahierten p-Wert zurück.
    """
    try:
        result = wildboottest(
            model,
            param=parameter_name,
            cluster=cluster_array,
            B=BOOTSTRAP_REPLICATIONS,
            bootstrap_type=bootstrap_type,
            impose_null=True,
            seed=SEED,
            show=False,
        )

        return extract_bootstrap_p_value(
            bootstrap_result=result,
            parameter_name=parameter_name,
        )

    except Exception as error:
        print(
            f"Bootstrap {bootstrap_type} für "
            f"'{parameter_name}' fehlgeschlagen: "
            f"{type(error).__name__}: {error}"
        )
        return np.nan


def holm_adjustment(
    p_values: pd.Series,
) -> pd.Series:
    """Berechnet Holm-korrigierte p-Werte."""
    valid = p_values.dropna()
    sorted_indices = valid.sort_values().index
    n_tests = len(valid)

    adjusted = pd.Series(
        np.nan,
        index=p_values.index,
        dtype=float,
    )

    previous_value = 0.0

    for rank, index in enumerate(
        sorted_indices,
        start=1,
    ):
        multiplier = n_tests - rank + 1

        adjusted_value = min(
            valid.loc[index] * multiplier,
            1.0,
        )

        adjusted_value = max(
            adjusted_value,
            previous_value,
        )

        adjusted.loc[index] = adjusted_value
        previous_value = adjusted_value

    return adjusted


def prepare_base_data(
    data: pd.DataFrame,
    all_outcomes: list[str],
) -> pd.DataFrame:
    """Filtert Mainstream-Parteien und bereitet Variablen auf."""
    required_columns = [
        AFD_SUPPORT,
        PARTY,
        PERIOD,
        MAINSTREAM,
        *all_outcomes,
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

    prepared = data.copy()

    prepared = prepared.loc[
        mainstream_mask(
            prepared[MAINSTREAM]
        )
    ].copy()

    for column in [
        AFD_SUPPORT,
        *all_outcomes,
    ]:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    prepared = prepared.dropna(
        subset=[
            AFD_SUPPORT,
            PARTY,
            PERIOD,
        ]
    ).copy()

    prepared[PARTY] = (
        prepared[PARTY]
        .astype(str)
        .str.strip()
        .str.lower()
        .astype("category")
    )

    prepared[PERIOD] = (
        prepared[PERIOD]
        .astype(str)
        .astype("category")
    )

    prepared["cluster_id"] = (
        prepared[PARTY]
        .cat.codes
        .astype(np.int64)
    )

    return prepared


def create_descriptives(
    analysis_df: pd.DataFrame,
    dimensions: dict,
    operationalization: str,
) -> pd.DataFrame:
    """Erstellt deskriptive Statistiken je Subdimension."""
    rows = []

    for dimension_key, outcome in dimensions.items():
        valid = analysis_df[outcome].dropna()

        rows.append(
            {
                "operationalization": (
                    operationalization
                ),
                "dimension": dimension_key,
                "dimension_label": (
                    DIMENSION_LABELS[
                        dimension_key
                    ]
                ),
                "outcome": outcome,
                "n": int(valid.count()),
                "mean": float(valid.mean()),
                "standard_deviation": float(
                    valid.std(ddof=1)
                ),
                "minimum": float(valid.min()),
                "median": float(valid.median()),
                "maximum": float(valid.max()),
            }
        )

    return pd.DataFrame(rows)


def run_dimension_models(
    analysis_df: pd.DataFrame,
    dimensions: dict,
    operationalization: str,
    expected_direction: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Schätzt ein Modell für jede der fünf Subdimensionen."""
    model_rows = []
    bootstrap_rows = []

    for dimension_key, outcome in dimensions.items():

        if USE_COMMON_COMPLETE_CASE_SAMPLE:
            dimension_df = analysis_df.copy()
        else:
            dimension_df = (
                analysis_df
                .dropna(subset=[outcome])
                .copy()
            )

        formula = f"""
            {outcome}
            ~ {CENTERED_SUPPORT}
            + C({PARTY})
            + C({PERIOD})
        """

        (
            model,
            ols_result,
            robust_result,
            cluster_array,
        ) = fit_cluster_model(
            formula=formula,
            data=dimension_df,
        )

        statistics = extract_parameter(
            ols_result=ols_result,
            robust_result=robust_result,
            parameter_name=CENTERED_SUPPORT,
        )

        p_wcr11 = run_wild_cluster_bootstrap(
            model=model,
            cluster_array=cluster_array,
            parameter_name=CENTERED_SUPPORT,
            bootstrap_type="11",
        )

        p_wcr31 = run_wild_cluster_bootstrap(
            model=model,
            cluster_array=cluster_array,
            parameter_name=CENTERED_SUPPORT,
            bootstrap_type="31",
        )

        direction_matches = (
            statistics["coefficient"] < 0
            if expected_direction == "negative"
            else statistics["coefficient"] > 0
        )

        model_rows.append(
            {
                "operationalization": (
                    operationalization
                ),
                "dimension": dimension_key,
                "dimension_label": (
                    DIMENSION_LABELS[
                        dimension_key
                    ]
                ),
                "outcome": outcome,
                "expected_direction": (
                    expected_direction
                ),
                **statistics,
                "wcr11_p_value": p_wcr11,
                "wcr31_p_value": p_wcr31,
                "direction_matches_expectation": (
                    direction_matches
                ),
                "significant_cluster_05": (
                    statistics["p_value"]
                    < ALPHA
                ),
                "n_observations": int(
                    ols_result.nobs
                ),
                "n_clusters": int(
                    dimension_df[
                        PARTY
                    ].nunique()
                ),
                "n_periods": int(
                    dimension_df[
                        PERIOD
                    ].nunique()
                ),
                "r_squared": float(
                    ols_result.rsquared
                ),
                "adjusted_r_squared": float(
                    ols_result.rsquared_adj
                ),
            }
        )

        bootstrap_rows.extend(
            [
                {
                    "operationalization": (
                        operationalization
                    ),
                    "dimension": dimension_key,
                    "dimension_label": (
                        DIMENSION_LABELS[
                            dimension_key
                        ]
                    ),
                    "bootstrap_type": "WCR11",
                    "p_value": p_wcr11,
                },
                {
                    "operationalization": (
                        operationalization
                    ),
                    "dimension": dimension_key,
                    "dimension_label": (
                        DIMENSION_LABELS[
                            dimension_key
                        ]
                    ),
                    "bootstrap_type": "WCR31",
                    "p_value": p_wcr31,
                },
            ]
        )

    results = pd.DataFrame(model_rows)

    results["p_value_holm"] = (
        holm_adjustment(
            results["p_value"]
        )
    )

    results["wcr11_p_value_holm"] = (
        holm_adjustment(
            results["wcr11_p_value"]
        )
    )

    results["wcr31_p_value_holm"] = (
        holm_adjustment(
            results["wcr31_p_value"]
        )
    )

    results["significant_cluster_holm_05"] = (
        results["p_value_holm"] < ALPHA
    )

    results["significant_wcr11_holm_05"] = (
        results["wcr11_p_value_holm"]
        < ALPHA
    )

    results["significant_wcr31_holm_05"] = (
        results["wcr31_p_value_holm"]
        < ALPHA
    )

    return (
        results,
        pd.DataFrame(bootstrap_rows),
    )


def run_pairwise_models(
    analysis_df: pd.DataFrame,
    dimensions: dict,
    operationalization: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vergleicht die AfD-Koeffizienten der Subdimensionen über
    paarweise Differenzmodelle.
    """
    pairwise_rows = []
    bootstrap_rows = []

    for dimension_a, dimension_b in combinations(
        dimensions.keys(),
        2,
    ):
        outcome_a = dimensions[dimension_a]
        outcome_b = dimensions[dimension_b]

        if USE_COMMON_COMPLETE_CASE_SAMPLE:
            pair_df = analysis_df.copy()
        else:
            pair_df = (
                analysis_df
                .dropna(
                    subset=[
                        outcome_a,
                        outcome_b,
                    ]
                )
                .copy()
            )

        difference_variable = (
            f"difference_{operationalization}_"
            f"{dimension_a}_{dimension_b}"
        )

        pair_df[difference_variable] = (
            pair_df[outcome_a]
            - pair_df[outcome_b]
        )

        formula = f"""
            {difference_variable}
            ~ {CENTERED_SUPPORT}
            + C({PARTY})
            + C({PERIOD})
        """

        (
            model,
            ols_result,
            robust_result,
            cluster_array,
        ) = fit_cluster_model(
            formula=formula,
            data=pair_df,
        )

        statistics = extract_parameter(
            ols_result=ols_result,
            robust_result=robust_result,
            parameter_name=CENTERED_SUPPORT,
        )

        p_wcr11 = run_wild_cluster_bootstrap(
            model=model,
            cluster_array=cluster_array,
            parameter_name=CENTERED_SUPPORT,
            bootstrap_type="11",
        )

        p_wcr31 = run_wild_cluster_bootstrap(
            model=model,
            cluster_array=cluster_array,
            parameter_name=CENTERED_SUPPORT,
            bootstrap_type="31",
        )

        comparison_label = (
            f"{DIMENSION_LABELS[dimension_a]} minus "
            f"{DIMENSION_LABELS[dimension_b]}"
        )

        pairwise_rows.append(
            {
                "operationalization": (
                    operationalization
                ),
                "dimension_a": dimension_a,
                "dimension_a_label": (
                    DIMENSION_LABELS[
                        dimension_a
                    ]
                ),
                "dimension_b": dimension_b,
                "dimension_b_label": (
                    DIMENSION_LABELS[
                        dimension_b
                    ]
                ),
                "comparison": comparison_label,
                "difference_outcome": (
                    difference_variable
                ),
                "slope_difference": (
                    statistics["coefficient"]
                ),
                "cluster_robust_se": (
                    statistics[
                        "cluster_robust_se"
                    ]
                ),
                "t_value": (
                    statistics["t_value"]
                ),
                "p_value": (
                    statistics["p_value"]
                ),
                "wcr11_p_value": p_wcr11,
                "wcr31_p_value": p_wcr31,
                "ci_95_low": (
                    statistics["ci_95_low"]
                ),
                "ci_95_high": (
                    statistics["ci_95_high"]
                ),
                "n_observations": int(
                    ols_result.nobs
                ),
                "n_clusters": int(
                    pair_df[PARTY].nunique()
                ),
                "n_periods": int(
                    pair_df[PERIOD].nunique()
                ),
            }
        )

        bootstrap_rows.extend(
            [
                {
                    "operationalization": (
                        operationalization
                    ),
                    "comparison": (
                        comparison_label
                    ),
                    "bootstrap_type": "WCR11",
                    "p_value": p_wcr11,
                },
                {
                    "operationalization": (
                        operationalization
                    ),
                    "comparison": (
                        comparison_label
                    ),
                    "bootstrap_type": "WCR31",
                    "p_value": p_wcr31,
                },
            ]
        )

    results = pd.DataFrame(
        pairwise_rows
    )

    results["p_value_holm"] = (
        holm_adjustment(
            results["p_value"]
        )
    )

    results["wcr11_p_value_holm"] = (
        holm_adjustment(
            results["wcr11_p_value"]
        )
    )

    results["wcr31_p_value_holm"] = (
        holm_adjustment(
            results["wcr31_p_value"]
        )
    )

    results["significant_cluster_holm_05"] = (
        results["p_value_holm"] < ALPHA
    )

    results["significant_wcr11_holm_05"] = (
        results["wcr11_p_value_holm"]
        < ALPHA
    )

    results["significant_wcr31_holm_05"] = (
        results["wcr31_p_value_holm"]
        < ALPHA
    )

    return (
        results,
        pd.DataFrame(bootstrap_rows),
    )


def run_operationalization(
    base_df: pd.DataFrame,
    operationalization: str,
    dimensions: dict,
    expected_direction: str,
) -> dict:
    """Führt alle Analysen für eine Operationalisierung aus."""
    if USE_COMMON_COMPLETE_CASE_SAMPLE:
        analysis_df = base_df.dropna(
            subset=list(dimensions.values())
        ).copy()
    else:
        analysis_df = base_df.copy()

    if analysis_df.empty:
        raise ValueError(
            f"Die Stichprobe für '{operationalization}' ist leer."
        )

    afd_support_mean = analysis_df[
        AFD_SUPPORT
    ].mean()

    analysis_df[CENTERED_SUPPORT] = (
        analysis_df[AFD_SUPPORT]
        - afd_support_mean
    )

    print("\n\n" + "#" * 78)
    print(
        f"RQ1: {operationalization.upper()}"
    )
    print("#" * 78)

    print(
        f"Partei-Intervalle: "
        f"{len(analysis_df)}"
    )
    print(
        f"Parteien/Cluster: "
        f"{analysis_df[PARTY].nunique()}"
    )
    print(
        f"Wahlperioden: "
        f"{analysis_df[PERIOD].nunique()}"
    )
    print(
        f"Mittelwert AfD-Unterstützung: "
        f"{afd_support_mean:.3f}"
    )

    descriptives = create_descriptives(
        analysis_df=analysis_df,
        dimensions=dimensions,
        operationalization=operationalization,
    )

    print("\nDeskriptive Statistiken:")
    print(
        descriptives[
            [
                "dimension_label",
                "n",
                "mean",
                "standard_deviation",
                "minimum",
                "median",
                "maximum",
            ]
        ]
        .round(DECIMALS)
        .to_string(index=False)
    )

    (
        dimension_results,
        dimension_bootstrap,
    ) = run_dimension_models(
        analysis_df=analysis_df,
        dimensions=dimensions,
        operationalization=operationalization,
        expected_direction=expected_direction,
    )

    print("\nDimensionsspezifische Modelle:")
    print(
        dimension_results[
            [
                "dimension_label",
                "coefficient",
                "cluster_robust_se",
                "t_value",
                "p_value",
                "p_value_holm",
                "wcr11_p_value",
                "wcr31_p_value",
                "ci_95_low",
                "ci_95_high",
                "n_observations",
            ]
        ]
        .round(DECIMALS)
        .to_string(index=False)
    )

    (
        pairwise_results,
        pairwise_bootstrap,
    ) = run_pairwise_models(
        analysis_df=analysis_df,
        dimensions=dimensions,
        operationalization=operationalization,
    )

    print("\nPaarweise Vergleiche:")
    print(
        pairwise_results[
            [
                "dimension_a_label",
                "dimension_b_label",
                "slope_difference",
                "cluster_robust_se",
                "p_value",
                "p_value_holm",
                "wcr11_p_value",
                "wcr11_p_value_holm",
                "wcr31_p_value",
                "wcr31_p_value_holm",
                "n_observations",
            ]
        ]
        .round(DECIMALS)
        .to_string(index=False)
    )

    prefix = (
        "rq1_distance"
        if operationalization == "distance"
        else "rq1_absolute"
    )

    save_result_dataframe(
        descriptives,
        OUTPUT_DIR
        / f"{prefix}_dimension_descriptives.csv",
    )

    save_result_dataframe(
        dimension_results,
        OUTPUT_DIR
        / f"{prefix}_dimension_models.csv",
    )

    save_result_dataframe(
        pairwise_results,
        OUTPUT_DIR
        / f"{prefix}_pairwise_dimension_tests.csv",
    )

    save_result_dataframe(
        dimension_bootstrap,
        OUTPUT_DIR
        / f"{prefix}_dimension_bootstrap.csv",
    )

    save_result_dataframe(
        pairwise_bootstrap,
        OUTPUT_DIR
        / f"{prefix}_pairwise_bootstrap.csv",
    )

    analysis_df.to_csv(
        OUTPUT_DIR
        / f"{prefix}_analysis_sample.csv",
        index=False,
        float_format="%.3f",
    )

    return {
        "descriptives": descriptives,
        "dimension_results": (
            dimension_results
        ),
        "pairwise_results": (
            pairwise_results
        ),
        "dimension_bootstrap": (
            dimension_bootstrap
        ),
        "pairwise_bootstrap": (
            pairwise_bootstrap
        ),
    }


# ============================================================
# 3. DATEN EINLESEN UND SPALTEN PRÜFEN
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Datei nicht gefunden:\n"
        f"{INPUT_FILE.resolve()}"
    )

df = pd.read_csv(
    INPUT_FILE
)

absolute_dimensions = ABSOLUTE_DIMENSIONS.copy()

print("=" * 78)
print("RQ1: GAL-TAN-Subdimensionen")
print("=" * 78)
print(f"Datei: {INPUT_FILE}")
print(f"Zeilen im Gesamtdatensatz: {len(df)}")
print(f"Spalten im Gesamtdatensatz: {len(df.columns)}")

print("\nVerwendete absolute Subdimensionsvariablen:")
for dimension, column in absolute_dimensions.items():
    print(
        f"- {DIMENSION_LABELS[dimension]}: "
        f"{column}"
    )

all_outcomes = list(
    dict.fromkeys(
        [
            *DISTANCE_DIMENSIONS.values(),
            *absolute_dimensions.values(),
        ]
    )
)

base_df = prepare_base_data(
    data=df,
    all_outcomes=all_outcomes,
)


# ============================================================
# 4. BEIDE OPERATIONALISIERUNGEN SCHÄTZEN
# ============================================================

distance_results = run_operationalization(
    base_df=base_df,
    operationalization="distance",
    dimensions=DISTANCE_DIMENSIONS,
    expected_direction="negative",
)

absolute_results = run_operationalization(
    base_df=base_df,
    operationalization="absolute",
    dimensions=absolute_dimensions,
    expected_direction="positive",
)


# ============================================================
# 5. GEMEINSAME ERGEBNISTABELLEN
# ============================================================

combined_descriptives = pd.concat(
    [
        distance_results["descriptives"],
        absolute_results["descriptives"],
    ],
    ignore_index=True,
)

combined_dimension_results = pd.concat(
    [
        distance_results[
            "dimension_results"
        ],
        absolute_results[
            "dimension_results"
        ],
    ],
    ignore_index=True,
)

combined_pairwise_results = pd.concat(
    [
        distance_results[
            "pairwise_results"
        ],
        absolute_results[
            "pairwise_results"
        ],
    ],
    ignore_index=True,
)


pairwise_report_table = create_pairwise_report_table(
    pairwise_results=combined_pairwise_results,
)

combined_dimension_bootstrap = pd.concat(
    [
        distance_results[
            "dimension_bootstrap"
        ],
        absolute_results[
            "dimension_bootstrap"
        ],
    ],
    ignore_index=True,
)

combined_pairwise_bootstrap = pd.concat(
    [
        distance_results[
            "pairwise_bootstrap"
        ],
        absolute_results[
            "pairwise_bootstrap"
        ],
    ],
    ignore_index=True,
)

save_result_dataframe(
    combined_descriptives,
    OUTPUT_DIR
    / "rq1_combined_dimension_descriptives.csv",
)

save_result_dataframe(
    combined_dimension_results,
    OUTPUT_DIR
    / "rq1_combined_dimension_models.csv",
)

save_result_dataframe(
    combined_pairwise_results,
    OUTPUT_DIR
    / "rq1_combined_pairwise_dimension_tests.csv",
)


save_result_dataframe(
    pairwise_report_table,
    OUTPUT_DIR
    / "rq1_table_all_pairwise_comparisons.csv",
)

try:
    pairwise_report_table.to_excel(
        OUTPUT_DIR
        / "rq1_table_all_pairwise_comparisons.xlsx",
        index=False,
        float_format="%.3f",
    )
except ImportError:
    print(
        "Excel-Datei konnte nicht erstellt werden. "
        "Installiere dafür openpyxl mit:\n"
        "pip install openpyxl"
    )

save_result_dataframe(
    combined_dimension_bootstrap,
    OUTPUT_DIR
    / "rq1_combined_dimension_bootstrap.csv",
)

save_result_dataframe(
    combined_pairwise_bootstrap,
    OUTPUT_DIR
    / "rq1_combined_pairwise_bootstrap.csv",
)


# ============================================================
# 6. ABSCHLUSSAUSGABE
# ============================================================

print("\n\n" + "=" * 78)
print("RQ1: ALLE ANALYSEN ABGESCHLOSSEN")
print("=" * 78)

print("\nDimensionsmodelle im Vergleich:")
print(
    combined_dimension_results[
        [
            "operationalization",
            "dimension_label",
            "coefficient",
            "cluster_robust_se",
            "p_value",
            "p_value_holm",
            "wcr11_p_value",
            "wcr31_p_value",
            "n_observations",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)

print("\nAlle paarweisen Vergleiche:")
print(
    pairwise_report_table
    .round(DECIMALS)
    .to_string(index=False)
)

print(
    "\nHinweis zur Interpretation:\n"
    "- Bei den Distanzmodellen bedeutet ein negativer "
    "Koeffizient eine abnehmende Distanz zur AfD.\n"
    "- Bei den absoluten Modellen bedeutet ein positiver "
    "Koeffizient eine Verschiebung in Richtung TAN, sofern "
    "höhere Werte den TAN-Pol darstellen.\n"
    "- Die paarweisen Differenzmodelle testen, ob sich die "
    "AfD-Koeffizienten zweier Subdimensionen unterscheiden.\n"
    "- Alle Modelle enthalten Partei- und "
    "Wahlperioden-Fixed-Effects."
)