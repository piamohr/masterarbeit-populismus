# ============================================================
# H1b: MODERATION DURCH UNTERSTÜTZUNGSVERLUSTE
#
# H1b:
# Je mehr Unterstützung eine Mainstream-Partei verloren hat,
# desto stärker ist der positionelle Annäherungseffekt.
#
# Die Hypothese wird mit zwei komplementären
# Operationalisierungen geprüft:
#
# MODELL A: Positionelle Distanz zur AfD
#   AV: gal_tan_distance_to_afd
#   Erwartung:
#   Interaktion afd_support_c × electoral_loss_c < 0
#
# MODELL B: Absolute GAL-TAN-Position
#   AV: mean_gal_tan_score
#   Erwartung:
#   Interaktion afd_support_c × electoral_loss_c > 0
#
# Voraussetzung für Modell B:
#   Höhere GAL-TAN-Werte stehen für stärker traditionalistische,
#   autoritäre und nationalistische Positionen.
#
# WICHTIG:
# party_support_change ist codiert als:
# aktueller Umfragewert minus vorheriger Umfragewert.
#
# Deshalb:
# electoral_loss = -party_support_change
#
# Positive Werte von electoral_loss bedeuten Verluste.
# Negative Werte bedeuten Gewinne.
#
# Für beide Modelle:
#   - Partei-Fixed-Effects
#   - Wahlperioden-Fixed-Effects
#   - Clusterrobuste Standardfehler auf Parteiebene
#   - Small-Sample-Korrektur
#   - Wild-Cluster-Bootstrap WCR11 und WCR31
#   - Simple-Slope-Analysen
#   - vorhergesagte Werte und Interaktionsgrafiken
#
# Alle Ergebniswerte werden mit drei Nachkommastellen
# ausgegeben und gespeichert.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from wildboottest.wildboottest import wildboottest


# ============================================================
# 1. EINSTELLUNGEN
# ============================================================

INPUT_FILE = Path(
    "data/processed/final_analysis/"
    "05_party_polling_interval_distances.csv"
)

OUTPUT_DIR = Path("results/hypothesis1b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AFD_SUPPORT = "afd_support_t"
SUPPORT_CHANGE = "party_support_change"
PARTY = "party_clean"
PERIOD = "wahlperiode_cat"
MAINSTREAM = "is_mainstream_party"

BOOTSTRAP_REPLICATIONS = 9999
SEED = 12345
ALPHA = 0.05
DECIMALS = 3

CENTERED_SUPPORT = "afd_support_c"
ELECTORAL_LOSS = "electoral_loss"
CENTERED_LOSS = "electoral_loss_c"
INTERACTION_NAME = "afd_x_loss"

MODEL_SPECS = {
    "distance": {
        "label": "Positionelle Distanz zur AfD",
        "outcome": "gal_tan_distance_to_afd",
        "expected_direction": "negative",
        "expected_sign": -1,
        "output_prefix": "h1b_distance",
        "prediction_column": "predicted_distance",
        "y_axis_label": "Vorhergesagte GAL–TAN-Distanz zur AfD",
        "figure_title": (
            "H1b: Unterstützungsverluste und positionelle Distanz zur AfD"
        ),
        "descriptive_variables": [
            "gal_tan_distance_to_afd",
            "mean_gal_tan_score",
            "afd_mean_gal_tan_score",
            "afd_support_t",
            "party_support_change",
        ],
    },
    "absolute": {
        "label": "Absolute GAL-TAN-Position",
        "outcome": "mean_gal_tan_score",
        "expected_direction": "positive",
        "expected_sign": 1,
        "output_prefix": "h1b_absolute_gal_tan",
        "prediction_column": "predicted_gal_tan",
        "y_axis_label": "Vorhergesagte absolute GAL–TAN-Position",
        "figure_title": (
            "H1b: Unterstützungsverluste und absolute GAL–TAN-Position"
        ),
        "descriptive_variables": [
            "mean_gal_tan_score",
            "gal_tan_distance_to_afd",
            "afd_mean_gal_tan_score",
            "afd_support_t",
            "party_support_change",
        ],
    },
}


# ============================================================
# 2. HILFSFUNKTIONEN
# ============================================================

def mainstream_mask(series: pd.Series) -> pd.Series:
    """
    Erzeugt einen robusten Filter für die Mainstream-Variable.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().any():
        return numeric.eq(1)

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "ja"])
    )


def save_bootstrap_result(
    result,
    output_file: Path,
) -> None:
    """
    Speichert ein wildboottest-Ergebnis mit drei
    Nachkommastellen als CSV.
    """
    if isinstance(result, pd.DataFrame):
        result.reset_index().to_csv(
            output_file,
            index=False,
            float_format="%.3f",
        )
    else:
        pd.DataFrame(
            {"result": [str(result)]}
        ).to_csv(
            output_file,
            index=False,
        )


def save_bootstrap_error(
    error: Exception,
    output_file: Path,
) -> None:
    """
    Speichert Fehlertyp und Fehlermeldung eines
    fehlgeschlagenen Bootstrap-Laufs.
    """
    pd.DataFrame(
        {
            "error_type": [type(error).__name__],
            "error_message": [str(error)],
        }
    ).to_csv(
        output_file,
        index=False,
    )


def save_result_dataframe(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    Speichert numerische Ergebniswerte mit drei
    Nachkommastellen.
    """
    dataframe.to_csv(
        output_file,
        index=False,
        float_format="%.3f",
    )


def direction_matches(
    coefficient: float,
    expected_sign: int,
) -> bool:
    """
    Prüft, ob ein Koeffizient in die erwartete Richtung weist.
    """
    if expected_sign == -1:
        return coefficient < 0

    if expected_sign == 1:
        return coefficient > 0

    raise ValueError(
        "expected_sign muss entweder -1 oder 1 sein."
    )


def interpret_interaction(
    coefficient: float,
    p_value: float,
    expected_sign: int,
    model_label: str,
) -> str:
    """
    Erstellt eine automatische Interpretation des
    Interaktionseffekts.
    """
    matches = direction_matches(
        coefficient=coefficient,
        expected_sign=expected_sign,
    )

    significant = p_value < ALPHA

    if matches and significant:
        return (
            f"{model_label}: Der Interaktionseffekt weist in die "
            "erwartete Richtung und ist statistisch signifikant."
        )

    if matches and not significant:
        return (
            f"{model_label}: Der Interaktionseffekt weist in die "
            "erwartete Richtung, ist aber nicht statistisch signifikant."
        )

    if not matches and significant:
        return (
            f"{model_label}: Der Interaktionseffekt ist statistisch "
            "signifikant, weist aber in die der Hypothese "
            "entgegengesetzte Richtung."
        )

    return (
        f"{model_label}: Der Interaktionseffekt weist nicht in die "
        "erwartete Richtung und ist nicht statistisch signifikant."
    )


def prepare_analysis_sample(
    data: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    """
    Erstellt die vollständige Mainstream-Analysestichprobe
    für eine abhängige Variable.
    """
    required_columns = [
        outcome,
        AFD_SUPPORT,
        SUPPORT_CHANGE,
        PARTY,
        PERIOD,
        MAINSTREAM,
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

    mask = (
        data[outcome].notna()
        & data[AFD_SUPPORT].notna()
        & data[SUPPORT_CHANGE].notna()
        & data[PARTY].notna()
        & data[PERIOD].notna()
        & data[MAINSTREAM].notna()
        & mainstream_mask(data[MAINSTREAM])
    )

    analysis_df = data.loc[mask].copy()

    for column in [
        outcome,
        AFD_SUPPORT,
        SUPPORT_CHANGE,
    ]:
        analysis_df[column] = pd.to_numeric(
            analysis_df[column],
            errors="coerce",
        )

    analysis_df = analysis_df.dropna(
        subset=[
            outcome,
            AFD_SUPPORT,
            SUPPORT_CHANGE,
            PARTY,
            PERIOD,
        ]
    ).copy()

    if analysis_df.empty:
        raise ValueError(
            f"Die Analysestichprobe für '{outcome}' ist leer."
        )

    analysis_df[PARTY] = (
        analysis_df[PARTY]
        .astype("category")
    )

    analysis_df[PERIOD] = (
        analysis_df[PERIOD]
        .astype("category")
    )

    analysis_df["cluster_id"] = (
        analysis_df[PARTY]
        .cat.codes
        .astype(np.int64)
    )

    if (analysis_df["cluster_id"] < 0).any():
        raise ValueError(
            "Bei der Erstellung der Cluster-ID sind "
            "ungültige Werte entstanden."
        )

    return analysis_df


def create_loss_variables(
    analysis_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float, float]:
    """
    Bildet die Verlustvariable, zentriert die Prädiktoren
    und erzeugt den Interaktionsterm.
    """
    centered_df = analysis_df.copy()

    centered_df[ELECTORAL_LOSS] = (
        -centered_df[SUPPORT_CHANGE]
    )

    afd_support_mean = centered_df[
        AFD_SUPPORT
    ].mean()

    electoral_loss_mean = centered_df[
        ELECTORAL_LOSS
    ].mean()

    electoral_loss_sd = centered_df[
        ELECTORAL_LOSS
    ].std(ddof=1)

    centered_df[CENTERED_SUPPORT] = (
        centered_df[AFD_SUPPORT]
        - afd_support_mean
    )

    centered_df[CENTERED_LOSS] = (
        centered_df[ELECTORAL_LOSS]
        - electoral_loss_mean
    )

    centered_df[INTERACTION_NAME] = (
        centered_df[CENTERED_SUPPORT]
        * centered_df[CENTERED_LOSS]
    )

    return (
        centered_df,
        float(afd_support_mean),
        float(electoral_loss_mean),
        float(electoral_loss_sd),
    )


def extract_parameter_result(
    ols_result,
    cluster_result,
    parameter_name: str,
) -> dict:
    """
    Extrahiert Koeffizient, clusterrobusten Standardfehler,
    t-Wert, p-Wert und 95%-Konfidenzintervall.
    """
    parameter_names = ols_result.model.exog_names

    if parameter_name not in parameter_names:
        raise ValueError(
            f"Parameter '{parameter_name}' wurde im Modell "
            "nicht gefunden."
        )

    parameter_index = parameter_names.index(
        parameter_name
    )

    confidence_interval = (
        cluster_result.conf_int()[parameter_index]
    )

    return {
        "coefficient": float(
            cluster_result.params[parameter_index]
        ),
        "cluster_robust_se": float(
            cluster_result.bse[parameter_index]
        ),
        "t_value": float(
            cluster_result.tvalues[parameter_index]
        ),
        "p_value": float(
            cluster_result.pvalues[parameter_index]
        ),
        "ci_95_low": float(
            confidence_interval[0]
        ),
        "ci_95_high": float(
            confidence_interval[1]
        ),
    }


def run_wild_cluster_bootstrap(
    model,
    cluster_array: np.ndarray,
    bootstrap_type: str,
    output_file: Path,
    heading: str,
):
    """
    Führt einen Wild-Cluster-Bootstrap für den
    Interaktionseffekt aus.
    """
    print("\n" + "=" * 72)
    print(heading)
    print("=" * 72)

    try:
        result = wildboottest(
            model,
            param=INTERACTION_NAME,
            cluster=cluster_array,
            B=BOOTSTRAP_REPLICATIONS,
            bootstrap_type=bootstrap_type,
            impose_null=True,
            seed=SEED,
            show=False,
        )

        if isinstance(result, pd.DataFrame):
            print(
                result.round(DECIMALS)
            )
        else:
            print(result)

        save_bootstrap_result(
            result=result,
            output_file=output_file,
        )

        return result

    except Exception as error:
        print(
            f"Der Wild-Cluster-Bootstrap {bootstrap_type} "
            "konnte nicht berechnet werden."
        )
        print(
            f"Fehlertyp: {type(error).__name__}"
        )
        print(
            f"Fehlermeldung: {error}"
        )

        save_bootstrap_error(
            error=error,
            output_file=output_file,
        )

        return None


def calculate_simple_slopes(
    ols_result,
    cluster_result,
    electoral_loss_mean: float,
    electoral_loss_sd: float,
) -> pd.DataFrame:
    """
    Berechnet den Effekt der AfD-Unterstützung bei niedrigem,
    mittlerem und hohem Unterstützungsverlust.
    """
    loss_levels = {
        "Niedriger Verlust / eher Gewinn": (
            electoral_loss_mean
            - electoral_loss_sd
        ),
        "Mittlerer Verlust": (
            electoral_loss_mean
        ),
        "Hoher Verlust": (
            electoral_loss_mean
            + electoral_loss_sd
        ),
    }

    parameter_names = (
        ols_result.model.exog_names
    )

    afd_index = parameter_names.index(
        CENTERED_SUPPORT
    )

    interaction_index = parameter_names.index(
        INTERACTION_NAME
    )

    simple_slope_rows = []

    for label, loss_value in loss_levels.items():
        loss_centered = (
            loss_value
            - electoral_loss_mean
        )

        restriction = np.zeros(
            len(parameter_names)
        )

        restriction[afd_index] = 1
        restriction[interaction_index] = (
            loss_centered
        )

        slope_test = cluster_result.t_test(
            restriction
        )

        slope = float(
            np.asarray(
                slope_test.effect
            ).squeeze()
        )

        slope_se = float(
            np.asarray(
                slope_test.sd
            ).squeeze()
        )

        slope_t = float(
            np.asarray(
                slope_test.tvalue
            ).squeeze()
        )

        slope_p = float(
            np.asarray(
                slope_test.pvalue
            ).squeeze()
        )

        slope_ci = np.asarray(
            slope_test.conf_int(
                alpha=ALPHA
            )
        ).squeeze()

        simple_slope_rows.append(
            {
                "loss_level": label,
                ELECTORAL_LOSS: loss_value,
                "effect_afd_support": slope,
                "cluster_robust_se": slope_se,
                "t_value": slope_t,
                "p_value": slope_p,
                "ci_95_low": float(
                    slope_ci[0]
                ),
                "ci_95_high": float(
                    slope_ci[1]
                ),
                "significant_05": (
                    slope_p < ALPHA
                ),
            }
        )

    return pd.DataFrame(
        simple_slope_rows
    )


def create_predictions(
    analysis_df: pd.DataFrame,
    ols_result,
    afd_support_mean: float,
    electoral_loss_mean: float,
    electoral_loss_sd: float,
    prediction_column: str,
) -> pd.DataFrame:
    """
    Erzeugt vorhergesagte Werte für niedrige, mittlere und
    hohe Unterstützungsverluste.
    """
    loss_levels = {
        "Niedriger Verlust / eher Gewinn": (
            electoral_loss_mean
            - electoral_loss_sd
        ),
        "Mittlerer Verlust": (
            electoral_loss_mean
        ),
        "Hoher Verlust": (
            electoral_loss_mean
            + electoral_loss_sd
        ),
    }

    afd_grid = np.linspace(
        analysis_df[AFD_SUPPORT].min(),
        analysis_df[AFD_SUPPORT].max(),
        100,
    )

    reference_party = (
        analysis_df[PARTY]
        .cat.categories[0]
    )

    reference_period = (
        analysis_df[PERIOD]
        .cat.categories[0]
    )

    prediction_frames = []

    for label, loss_value in loss_levels.items():
        prediction_data = pd.DataFrame(
            {
                AFD_SUPPORT: afd_grid,
                ELECTORAL_LOSS: loss_value,
                PARTY: reference_party,
                PERIOD: reference_period,
            }
        )

        prediction_data[CENTERED_SUPPORT] = (
            prediction_data[AFD_SUPPORT]
            - afd_support_mean
        )

        prediction_data[CENTERED_LOSS] = (
            prediction_data[ELECTORAL_LOSS]
            - electoral_loss_mean
        )

        prediction_data[INTERACTION_NAME] = (
            prediction_data[CENTERED_SUPPORT]
            * prediction_data[CENTERED_LOSS]
        )

        prediction = (
            ols_result
            .get_prediction(prediction_data)
            .summary_frame(alpha=ALPHA)
        )

        prediction_data["loss_level"] = label

        prediction_data[prediction_column] = (
            prediction["mean"]
        )

        prediction_data["ci_95_low"] = (
            prediction["mean_ci_lower"]
        )

        prediction_data["ci_95_high"] = (
            prediction["mean_ci_upper"]
        )

        prediction_frames.append(
            prediction_data
        )

    return pd.concat(
        prediction_frames,
        ignore_index=True,
    )


def create_interaction_plot(
    predictions: pd.DataFrame,
    prediction_column: str,
    y_axis_label: str,
    figure_title: str,
    output_file: Path,
) -> None:
    """
    Speichert eine Interaktionsgrafik.
    """
    plt.figure(
        figsize=(9, 6)
    )

    for label, group in predictions.groupby(
        "loss_level",
        sort=False,
    ):
        plt.plot(
            group[AFD_SUPPORT],
            group[prediction_column],
            label=label,
        )

    plt.xlabel(
        "AfD-Unterstützung in Prozentpunkten"
    )

    plt.ylabel(
        y_axis_label
    )

    plt.title(
        figure_title
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()


def run_model_specification(
    data: pd.DataFrame,
    model_key: str,
    model_spec: dict,
) -> dict:
    """
    Führt das vollständige H1b-Modell für eine
    Operationalisierung aus.
    """
    label = model_spec["label"]
    outcome = model_spec["outcome"]
    expected_direction = (
        model_spec["expected_direction"]
    )
    expected_sign = (
        model_spec["expected_sign"]
    )
    prefix = (
        model_spec["output_prefix"]
    )
    prediction_column = (
        model_spec["prediction_column"]
    )

    output_model = (
        OUTPUT_DIR
        / f"{prefix}_model_results.csv"
    )

    output_bootstrap_11 = (
        OUTPUT_DIR
        / f"{prefix}_bootstrap_wcr11.csv"
    )

    output_bootstrap_31 = (
        OUTPUT_DIR
        / f"{prefix}_bootstrap_wcr31.csv"
    )

    output_simple_slopes = (
        OUTPUT_DIR
        / f"{prefix}_simple_slopes.csv"
    )

    output_predictions = (
        OUTPUT_DIR
        / f"{prefix}_predicted_values.csv"
    )

    output_figure = (
        OUTPUT_DIR
        / f"{prefix}_interaction_plot.png"
    )

    output_sample = (
        OUTPUT_DIR
        / f"{prefix}_analysis_sample.csv"
    )

    analysis_df = prepare_analysis_sample(
        data=data,
        outcome=outcome,
    )

    (
        analysis_df,
        afd_support_mean,
        electoral_loss_mean,
        electoral_loss_sd,
    ) = create_loss_variables(
        analysis_df=analysis_df,
    )

    cluster_array = (
        analysis_df["cluster_id"]
        .to_numpy(dtype=np.int64)
    )

    print("\n\n" + "#" * 72)
    print(
        f"STARTE H1b-MODELL: "
        f"{label.upper()}"
    )
    print("#" * 72)

    print(
        f"Analyseeinheiten: "
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

    print("\nOriginale Veränderungsvariable:")
    print(
        analysis_df[SUPPORT_CHANGE]
        .describe()
        .round(DECIMALS)
    )

    print("\nNeu gebildete Verlustvariable:")
    print(
        analysis_df[ELECTORAL_LOSS]
        .describe()
        .round(DECIMALS)
    )

    print("\nInterpretation der Verlustvariable:")
    print("- positiver Wert = Unterstützungsverlust")
    print("- null           = keine Veränderung")
    print("- negativer Wert = Unterstützungsgewinn")

    print("\nAnteile:")
    print(
        pd.Series(
            {
                "Verlust": (
                    analysis_df[ELECTORAL_LOSS] > 0
                ).mean(),
                "Unverändert": (
                    analysis_df[ELECTORAL_LOSS] == 0
                ).mean(),
                "Gewinn": (
                    analysis_df[ELECTORAL_LOSS] < 0
                ).mean(),
            }
        ).round(DECIMALS)
    )

    print("\nMittelwerte für die Zentrierung:")
    print(
        f"AfD-Unterstützung: "
        f"{afd_support_mean:.3f}"
    )
    print(
        f"Unterstützungsverlust: "
        f"{electoral_loss_mean:.3f}"
    )
    print(
        f"SD Unterstützungsverlust: "
        f"{electoral_loss_sd:.3f}"
    )

    descriptive_variables = model_spec["descriptive_variables"]

    print("\nDeskriptive Werte:")
    print(
        analysis_df[descriptive_variables]
        .describe()
        .round(DECIMALS)
    )

    # --------------------------------------------------------
    # Interaktionsmodell mit Partei- und Wahlperioden-FE
    # --------------------------------------------------------

    formula = f"""
        {outcome}
        ~ {CENTERED_SUPPORT}
        + {CENTERED_LOSS}
        + {INTERACTION_NAME}
        + C({PARTY})
        + C({PERIOD})
    """

    model = smf.ols(
        formula=formula,
        data=analysis_df,
    )

    ols_result = model.fit()

    if len(cluster_array) != int(
        ols_result.nobs
    ):
        raise ValueError(
            "Die Zahl der Clusterkennungen stimmt nicht mit "
            "der Zahl der Modellbeobachtungen überein."
        )

    cluster_result = (
        ols_result
        .get_robustcov_results(
            cov_type="cluster",
            groups=cluster_array,
            use_correction=True,
            df_correction=True,
            use_t=True,
        )
    )

    interaction_stats = extract_parameter_result(
        ols_result=ols_result,
        cluster_result=cluster_result,
        parameter_name=INTERACTION_NAME,
    )

    supported = (
        direction_matches(
            coefficient=interaction_stats[
                "coefficient"
            ],
            expected_sign=expected_sign,
        )
        and interaction_stats["p_value"] < ALPHA
    )

    interpretation = interpret_interaction(
        coefficient=interaction_stats[
            "coefficient"
        ],
        p_value=interaction_stats[
            "p_value"
        ],
        expected_sign=expected_sign,
        model_label=label,
    )

    print("\nModellspezifikation:")
    print(
        formula.strip()
    )

    print("\n" + "=" * 72)
    print(
        f"H1b: Interaktionsmodell – {label}"
    )
    print("=" * 72)

    print(
        "Interaktionskoeffizient: "
        f"{interaction_stats['coefficient']:.3f}"
    )
    print(
        "Clusterrobuster SE:      "
        f"{interaction_stats['cluster_robust_se']:.3f}"
    )
    print(
        "t-Wert:                  "
        f"{interaction_stats['t_value']:.3f}"
    )
    print(
        "p-Wert:                  "
        f"{interaction_stats['p_value']:.3f}"
    )
    print(
        "95%-KI:                  "
        f"[{interaction_stats['ci_95_low']:.3f}, "
        f"{interaction_stats['ci_95_high']:.3f}]"
    )
    print(
        "H1b unterstützt:         "
        f"{'Ja' if supported else 'Nein'}"
    )

    print("\nInterpretation:")
    print(
        interpretation
    )

    # --------------------------------------------------------
    # Wild-Cluster-Bootstrap
    # --------------------------------------------------------

    run_wild_cluster_bootstrap(
        model=model,
        cluster_array=cluster_array,
        bootstrap_type="11",
        output_file=output_bootstrap_11,
        heading=f"H1b: WCR11 – {label}",
    )

    run_wild_cluster_bootstrap(
        model=model,
        cluster_array=cluster_array,
        bootstrap_type="31",
        output_file=output_bootstrap_31,
        heading=f"H1b: WCR31 – {label}",
    )

    # --------------------------------------------------------
    # Simple Slopes
    # --------------------------------------------------------

    simple_slopes = calculate_simple_slopes(
        ols_result=ols_result,
        cluster_result=cluster_result,
        electoral_loss_mean=electoral_loss_mean,
        electoral_loss_sd=electoral_loss_sd,
    )

    print("\n" + "=" * 72)
    print(
        f"Simple Slopes – {label}"
    )
    print("=" * 72)

    print(
        simple_slopes
        .round(DECIMALS)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Vorhergesagte Werte und Grafik
    # --------------------------------------------------------

    predictions = create_predictions(
        analysis_df=analysis_df,
        ols_result=ols_result,
        afd_support_mean=afd_support_mean,
        electoral_loss_mean=electoral_loss_mean,
        electoral_loss_sd=electoral_loss_sd,
        prediction_column=prediction_column,
    )

    create_interaction_plot(
        predictions=predictions,
        prediction_column=prediction_column,
        y_axis_label=model_spec[
            "y_axis_label"
        ],
        figure_title=model_spec[
            "figure_title"
        ],
        output_file=output_figure,
    )

    # --------------------------------------------------------
    # Ergebnisse speichern
    # --------------------------------------------------------

    model_output = pd.DataFrame(
        [
            {
                "hypothesis": "H1b",
                "operationalization": model_key,
                "model_label": label,
                "model": (
                    "party_and_period_fixed_effects"
                ),
                "outcome": outcome,
                "predictor": AFD_SUPPORT,
                "moderator": ELECTORAL_LOSS,
                "interaction": INTERACTION_NAME,
                "expected_direction": (
                    expected_direction
                ),
                **interaction_stats,
                "n_observations": int(
                    ols_result.nobs
                ),
                "n_clusters": int(
                    analysis_df[PARTY].nunique()
                ),
                "n_periods": int(
                    analysis_df[PERIOD].nunique()
                ),
                "r_squared": float(
                    ols_result.rsquared
                ),
                "adjusted_r_squared": float(
                    ols_result.rsquared_adj
                ),
                "afd_support_mean": (
                    afd_support_mean
                ),
                "electoral_loss_mean": (
                    electoral_loss_mean
                ),
                "electoral_loss_sd": (
                    electoral_loss_sd
                ),
                "supported": supported,
                "interpretation": (
                    interpretation
                ),
            }
        ]
    )

    save_result_dataframe(
        dataframe=model_output,
        output_file=output_model,
    )

    save_result_dataframe(
        dataframe=simple_slopes,
        output_file=output_simple_slopes,
    )

    save_result_dataframe(
        dataframe=predictions,
        output_file=output_predictions,
    )

    analysis_df.to_csv(
        output_sample,
        index=False,
        float_format="%.3f",
    )

    print("\nGespeicherte Dateien:")
    print(f"- {output_model}")
    print(f"- {output_bootstrap_11}")
    print(f"- {output_bootstrap_31}")
    print(f"- {output_simple_slopes}")
    print(f"- {output_predictions}")
    print(f"- {output_figure}")
    print(f"- {output_sample}")

    return {
        "main": model_output,
        "simple_slopes": simple_slopes,
    }


# ============================================================
# 3. DATEN EINLESEN
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Datei nicht gefunden:\n"
        f"{INPUT_FILE.resolve()}"
    )

df = pd.read_csv(
    INPUT_FILE
)

print("=" * 72)
print("H1b: Moderation durch Unterstützungsverluste")
print("=" * 72)
print(f"Datei: {INPUT_FILE}")
print(f"Zeilen im Gesamtdatensatz: {len(df)}")
print(f"Spalten im Gesamtdatensatz: {len(df.columns)}")


# ============================================================
# 4. BEIDE H1b-OPERATIONALISIERUNGEN SCHÄTZEN
# ============================================================

all_main_results = []
all_simple_slopes = []

for model_key, model_spec in MODEL_SPECS.items():
    results = run_model_specification(
        data=df,
        model_key=model_key,
        model_spec=model_spec,
    )

    all_main_results.append(
        results["main"]
    )

    slopes_with_model = (
        results["simple_slopes"]
        .copy()
    )

    slopes_with_model.insert(
        0,
        "operationalization",
        model_key,
    )

    slopes_with_model.insert(
        1,
        "model_label",
        model_spec["label"],
    )

    all_simple_slopes.append(
        slopes_with_model
    )


# ============================================================
# 5. GEMEINSAME ERGEBNISTABELLEN
# ============================================================

combined_main_results = pd.concat(
    all_main_results,
    ignore_index=True,
)

combined_simple_slopes = pd.concat(
    all_simple_slopes,
    ignore_index=True,
)

OUTPUT_COMBINED_MODELS = (
    OUTPUT_DIR
    / "h1b_combined_models.csv"
)

OUTPUT_COMBINED_SIMPLE_SLOPES = (
    OUTPUT_DIR
    / "h1b_combined_simple_slopes.csv"
)

save_result_dataframe(
    dataframe=combined_main_results,
    output_file=OUTPUT_COMBINED_MODELS,
)

save_result_dataframe(
    dataframe=combined_simple_slopes,
    output_file=OUTPUT_COMBINED_SIMPLE_SLOPES,
)


# ============================================================
# 6. ABSCHLUSSAUSGABE
# ============================================================

print("\n\n" + "=" * 72)
print("H1b: ALLE ANALYSEN ABGESCHLOSSEN")
print("=" * 72)

print("\nInteraktionsmodelle im Vergleich:")
print(
    combined_main_results[
        [
            "model_label",
            "outcome",
            "expected_direction",
            "coefficient",
            "cluster_robust_se",
            "t_value",
            "p_value",
            "ci_95_low",
            "ci_95_high",
            "n_observations",
            "n_clusters",
            "supported",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)

print("\nGemeinsame Ergebnisdateien:")
print(
    f"- Interaktionsmodelle: "
    f"{OUTPUT_COMBINED_MODELS}"
)
print(
    f"- Simple Slopes:       "
    f"{OUTPUT_COMBINED_SIMPLE_SLOPES}"
)

print(
    "\nHinweis zur Interpretation:\n"
    "- Beim Distanzmodell unterstützt ein negativer "
    "Interaktionskoeffizient H1b.\n"
    "- Beim Modell der absoluten GAL-TAN-Position unterstützt "
    "ein positiver Interaktionskoeffizient H1b, sofern höhere "
    "Werte den TAN-Pol abbilden.\n"
    "- Positive Werte der Variable electoral_loss bedeuten "
    "Unterstützungsverluste; negative Werte bedeuten Gewinne."
)