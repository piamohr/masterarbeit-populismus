# ============================================================
# H1a: MODERATION DURCH DIE IDEOLOGISCHE NÄHE ZUR AFD
#
# H1a:
# Je näher eine Mainstream-Partei der AfD steht, desto stärker
# ist der positionelle Annäherungseffekt.
#
# MODELL A: Positionelle Distanz zur AfD
#   AV: gal_tan_distance_to_afd
#   Erwartung: afd_support_c × proximity_c < 0
#
# MODELL B: Absolute GAL-TAN-Position
#   AV: mean_gal_tan_score
#   Erwartung: afd_support_c × proximity_c > 0
#
# Voraussetzung für Modell B:
# Höhere GAL-TAN-Werte stehen für stärker traditionalistische,
# autoritäre und nationalistische Positionen.
#
# Für beide Modelle:
#   - Partei-Fixed-Effects
#   - Wahlperioden-Fixed-Effects
#   - Clusterrobuste Standardfehler auf Parteiebene
#   - Small-Sample-Korrektur
#   - Wild-Cluster-Bootstrap WCR11 und WCR31
#   - Simple-Slope-Analysen
#   - Vorhersagen und Interaktionsgrafiken
#
# Der zeitkonstante Haupteffekt der Nähe wird durch die
# Partei-Fixed-Effects absorbiert und nicht separat geschätzt.
# Alle Ergebniswerte werden mit drei Nachkommastellen ausgegeben.
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
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

OUTPUT_DIR = Path("results/hypothesis1a")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AFD_SUPPORT = "afd_support_t"
MODERATOR = "afd_proximity_ordinal"
PARTY = "party_clean"
PERIOD = "wahlperiode_cat"
MAINSTREAM = "is_mainstream_party"

CENTERED_SUPPORT = "afd_support_c"
CENTERED_PROXIMITY = "proximity_c"
INTERACTION = "afd_x_proximity"

BOOTSTRAP_REPLICATIONS = 9999
SEED = 12345
ALPHA = 0.05
DECIMALS = 3

MODEL_SPECS = {
    "distance": {
        "label": "Positionelle Distanz zur AfD",
        "outcome": "gal_tan_distance_to_afd",
        "expected_direction": "negative",
        "expected_sign": -1,
        "output_prefix": "h1a_distance",
        "prediction_column": "predicted_distance",
        "y_axis_label": "Vorhergesagte GAL–TAN-Distanz zur AfD",
        "figure_title": (
            "H1a: Ideologische Nähe und positionelle Distanz zur AfD"
        ),
    },
    "absolute": {
        "label": "Absolute GAL-TAN-Position",
        "outcome": "mean_gal_tan_score",
        "expected_direction": "positive",
        "expected_sign": 1,
        "output_prefix": "h1a_absolute_gal_tan",
        "prediction_column": "predicted_gal_tan",
        "y_axis_label": "Vorhergesagte absolute GAL–TAN-Position",
        "figure_title": (
            "H1a: Ideologische Nähe und absolute GAL–TAN-Position"
        ),
    },
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


def save_dataframe(dataframe: pd.DataFrame, output_file: Path) -> None:
    """Speichert numerische Werte mit drei Nachkommastellen."""
    dataframe.to_csv(
        output_file,
        index=False,
        float_format="%.3f",
    )


def save_bootstrap_result(result, output_file: Path) -> None:
    """Speichert Wild-Cluster-Bootstrap-Ergebnisse."""
    if isinstance(result, pd.DataFrame):
        result.reset_index().to_csv(
            output_file,
            index=False,
            float_format="%.3f",
        )
    else:
        pd.DataFrame({"result": [str(result)]}).to_csv(
            output_file,
            index=False,
        )


def save_bootstrap_error(error: Exception, output_file: Path) -> None:
    """Speichert Fehlertyp und Fehlermeldung."""
    pd.DataFrame(
        {
            "error_type": [type(error).__name__],
            "error_message": [str(error)],
        }
    ).to_csv(output_file, index=False)


def direction_matches(coefficient: float, expected_sign: int) -> bool:
    """Prüft, ob der Interaktionseffekt erwartungsgemäß ausfällt."""
    if expected_sign == -1:
        return coefficient < 0
    if expected_sign == 1:
        return coefficient > 0
    raise ValueError("expected_sign muss -1 oder 1 sein.")


def interpret_interaction(
    coefficient: float,
    p_value: float,
    expected_sign: int,
    label: str,
) -> str:
    """Erstellt eine richtungsbezogene Interpretation."""
    matches = direction_matches(coefficient, expected_sign)
    significant = p_value < ALPHA

    if matches and significant:
        return (
            f"{label}: Der Interaktionseffekt weist in die erwartete "
            "Richtung und ist statistisch signifikant."
        )
    if matches:
        return (
            f"{label}: Der Interaktionseffekt weist in die erwartete "
            "Richtung, ist aber nicht statistisch signifikant."
        )
    if significant:
        return (
            f"{label}: Der Interaktionseffekt ist statistisch "
            "signifikant, weist aber in die entgegengesetzte Richtung."
        )
    return (
        f"{label}: Der Interaktionseffekt weist nicht in die erwartete "
        "Richtung und ist nicht statistisch signifikant."
    )


def prepare_analysis_sample(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Erstellt die Mainstream-Analysestichprobe."""
    required_columns = [
        outcome,
        AFD_SUPPORT,
        MODERATOR,
        PARTY,
        PERIOD,
        MAINSTREAM,
    ]

    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Folgende Variablen fehlen:\n"
            + "\n".join(f"- {column}" for column in missing_columns)
        )

    mask = (
        data[outcome].notna()
        & data[AFD_SUPPORT].notna()
        & data[MODERATOR].notna()
        & data[PARTY].notna()
        & data[PERIOD].notna()
        & data[MAINSTREAM].notna()
        & mainstream_mask(data[MAINSTREAM])
    )

    analysis_df = data.loc[mask].copy()

    for column in [outcome, AFD_SUPPORT, MODERATOR]:
        analysis_df[column] = pd.to_numeric(
            analysis_df[column],
            errors="coerce",
        )

    analysis_df = analysis_df.dropna(
        subset=[outcome, AFD_SUPPORT, MODERATOR, PARTY, PERIOD]
    ).copy()

    if analysis_df.empty:
        raise ValueError(f"Die Analysestichprobe für '{outcome}' ist leer.")

    analysis_df[PARTY] = analysis_df[PARTY].astype("category")
    analysis_df[PERIOD] = analysis_df[PERIOD].astype("category")
    analysis_df["cluster_id"] = (
        analysis_df[PARTY].cat.codes.astype(np.int64)
    )

    if (analysis_df["cluster_id"] < 0).any():
        raise ValueError("Ungültige Cluster-IDs entstanden.")

    return analysis_df


def center_variables(
    analysis_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """Zentriert Prädiktor und Moderator und bildet die Interaktion."""
    analysis_df = analysis_df.copy()
    afd_support_mean = float(analysis_df[AFD_SUPPORT].mean())
    proximity_mean = float(analysis_df[MODERATOR].mean())

    analysis_df[CENTERED_SUPPORT] = (
        analysis_df[AFD_SUPPORT] - afd_support_mean
    )
    analysis_df[CENTERED_PROXIMITY] = (
        analysis_df[MODERATOR] - proximity_mean
    )
    analysis_df[INTERACTION] = (
        analysis_df[CENTERED_SUPPORT]
        * analysis_df[CENTERED_PROXIMITY]
    )

    return analysis_df, afd_support_mean, proximity_mean


def extract_parameter_result(
    ols_result,
    cluster_result,
    parameter_name: str,
) -> dict:
    """Extrahiert den interessierenden Modellparameter."""
    parameter_names = ols_result.model.exog_names
    if parameter_name not in parameter_names:
        raise ValueError(f"Parameter '{parameter_name}' fehlt im Modell.")

    parameter_index = parameter_names.index(parameter_name)
    ci = cluster_result.conf_int()[parameter_index]

    return {
        "coefficient": float(cluster_result.params[parameter_index]),
        "cluster_robust_se": float(cluster_result.bse[parameter_index]),
        "t_value": float(cluster_result.tvalues[parameter_index]),
        "p_value": float(cluster_result.pvalues[parameter_index]),
        "ci_95_low": float(ci[0]),
        "ci_95_high": float(ci[1]),
    }


def run_wild_cluster_bootstrap(
    model,
    cluster_array: np.ndarray,
    bootstrap_type: str,
    output_file: Path,
    heading: str,
):
    """Führt den Wild-Cluster-Bootstrap für die Interaktion aus."""
    print("\n" + "=" * 72)
    print(heading)
    print("=" * 72)

    try:
        result = wildboottest(
            model,
            param=INTERACTION,
            cluster=cluster_array,
            B=BOOTSTRAP_REPLICATIONS,
            bootstrap_type=bootstrap_type,
            impose_null=True,
            seed=SEED,
            show=False,
        )

        if isinstance(result, pd.DataFrame):
            print(result.round(DECIMALS))
        else:
            print(result)

        save_bootstrap_result(result, output_file)
        return result

    except Exception as error:
        print(
            f"Der Wild-Cluster-Bootstrap {bootstrap_type} "
            "konnte nicht berechnet werden."
        )
        print(f"Fehlertyp: {type(error).__name__}")
        print(f"Fehlermeldung: {error}")
        save_bootstrap_error(error, output_file)
        return None


def calculate_simple_slopes(
    analysis_df: pd.DataFrame,
    ols_result,
    cluster_result,
    proximity_mean: float,
) -> pd.DataFrame:
    """Berechnet den AfD-Unterstützungseffekt je Nähewert."""
    parameter_names = ols_result.model.exog_names
    afd_index = parameter_names.index(CENTERED_SUPPORT)
    interaction_index = parameter_names.index(INTERACTION)

    rows = []
    for proximity_value in sorted(analysis_df[MODERATOR].unique()):
        proximity_centered = proximity_value - proximity_mean

        restriction = np.zeros(len(parameter_names))
        restriction[afd_index] = 1
        restriction[interaction_index] = proximity_centered

        slope_test = cluster_result.t_test(restriction)
        slope_ci = np.asarray(
            slope_test.conf_int(alpha=ALPHA)
        ).squeeze()

        slope = float(np.asarray(slope_test.effect).squeeze())
        slope_se = float(np.asarray(slope_test.sd).squeeze())
        slope_t = float(np.asarray(slope_test.tvalue).squeeze())
        slope_p = float(np.asarray(slope_test.pvalue).squeeze())

        rows.append(
            {
                MODERATOR: float(proximity_value),
                "effect_afd_support": slope,
                "cluster_robust_se": slope_se,
                "t_value": slope_t,
                "p_value": slope_p,
                "ci_95_low": float(slope_ci[0]),
                "ci_95_high": float(slope_ci[1]),
                "significant_05": slope_p < ALPHA,
            }
        )

    return pd.DataFrame(rows)


def create_predictions(
    analysis_df: pd.DataFrame,
    ols_result,
    afd_support_mean: float,
    proximity_mean: float,
    prediction_column: str,
) -> pd.DataFrame:
    """Erzeugt Vorhersagen für alle beobachteten Nähewerte."""
    afd_grid = np.linspace(
        analysis_df[AFD_SUPPORT].min(),
        analysis_df[AFD_SUPPORT].max(),
        100,
    )

    reference_party = analysis_df[PARTY].cat.categories[0]
    reference_period = analysis_df[PERIOD].cat.categories[0]
    frames = []

    for proximity_value in sorted(analysis_df[MODERATOR].unique()):
        prediction_data = pd.DataFrame(
            {
                AFD_SUPPORT: afd_grid,
                MODERATOR: proximity_value,
                PARTY: reference_party,
                PERIOD: reference_period,
            }
        )

        prediction_data[CENTERED_SUPPORT] = (
            prediction_data[AFD_SUPPORT] - afd_support_mean
        )
        prediction_data[CENTERED_PROXIMITY] = (
            prediction_data[MODERATOR] - proximity_mean
        )
        prediction_data[INTERACTION] = (
            prediction_data[CENTERED_SUPPORT]
            * prediction_data[CENTERED_PROXIMITY]
        )

        prediction = ols_result.get_prediction(
            prediction_data
        ).summary_frame(alpha=ALPHA)

        prediction_data[prediction_column] = prediction["mean"]
        prediction_data["ci_95_low"] = prediction["mean_ci_lower"]
        prediction_data["ci_95_high"] = prediction["mean_ci_upper"]
        frames.append(prediction_data)

    return pd.concat(frames, ignore_index=True)


def create_interaction_plot(
    predictions: pd.DataFrame,
    prediction_column: str,
    y_axis_label: str,
    figure_title: str,
    output_file: Path,
) -> None:
    """Speichert die Interaktionsgrafik."""
    plt.figure(figsize=(9, 6))

    for proximity_value, group in predictions.groupby(MODERATOR):
        plt.plot(
            group[AFD_SUPPORT],
            group[prediction_column],
            label=f"AfD-Nähe = {int(proximity_value)}",
        )

    plt.xlabel("AfD-Unterstützung in Prozentpunkten")
    plt.ylabel(y_axis_label)
    plt.title(figure_title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def run_model_specification(
    data: pd.DataFrame,
    model_key: str,
    model_spec: dict,
) -> dict:
    """Führt eine vollständige H1a-Operationalisierung aus."""
    label = model_spec["label"]
    outcome = model_spec["outcome"]
    expected_direction = model_spec["expected_direction"]
    expected_sign = model_spec["expected_sign"]
    prefix = model_spec["output_prefix"]
    prediction_column = model_spec["prediction_column"]

    output_model = OUTPUT_DIR / f"{prefix}_model_results.csv"
    output_bootstrap_11 = OUTPUT_DIR / f"{prefix}_bootstrap_wcr11.csv"
    output_bootstrap_31 = OUTPUT_DIR / f"{prefix}_bootstrap_wcr31.csv"
    output_simple_slopes = OUTPUT_DIR / f"{prefix}_simple_slopes.csv"
    output_predictions = OUTPUT_DIR / f"{prefix}_predicted_values.csv"
    output_figure = OUTPUT_DIR / f"{prefix}_interaction_plot.png"
    output_sample = OUTPUT_DIR / f"{prefix}_analysis_sample.csv"

    analysis_df = prepare_analysis_sample(data, outcome)
    analysis_df, afd_support_mean, proximity_mean = center_variables(
        analysis_df
    )
    cluster_array = analysis_df["cluster_id"].to_numpy(dtype=np.int64)

    print("\n\n" + "#" * 72)
    print(f"STARTE H1a-MODELL: {label.upper()}")
    print("#" * 72)
    print(f"Analyseeinheiten: {len(analysis_df)}")
    print(f"Parteien/Cluster: {analysis_df[PARTY].nunique()}")
    print(f"Wahlperioden: {analysis_df[PERIOD].nunique()}")

    print("\nNähewerte nach Partei:")
    print(
        analysis_df[[PARTY, MODERATOR]]
        .drop_duplicates()
        .sort_values(MODERATOR)
        .to_string(index=False)
    )

    print("\nMittelwerte für die Zentrierung:")
    print(f"AfD-Unterstützung: {afd_support_mean:.3f}")
    print(f"AfD-Nähe:          {proximity_mean:.3f}")

    formula = f"""
        {outcome}
        ~ {CENTERED_SUPPORT}
        + {INTERACTION}
        + C({PARTY})
        + C({PERIOD})
    """

    model = smf.ols(formula=formula, data=analysis_df)
    ols_result = model.fit()

    if len(cluster_array) != int(ols_result.nobs):
        raise ValueError(
            "Cluster-Array und Modellbeobachtungen stimmen nicht überein."
        )

    cluster_result = ols_result.get_robustcov_results(
        cov_type="cluster",
        groups=cluster_array,
        use_correction=True,
        df_correction=True,
        use_t=True,
    )

    stats = extract_parameter_result(
        ols_result,
        cluster_result,
        INTERACTION,
    )

    supported = (
        direction_matches(stats["coefficient"], expected_sign)
        and stats["p_value"] < ALPHA
    )

    interpretation = interpret_interaction(
        stats["coefficient"],
        stats["p_value"],
        expected_sign,
        label,
    )

    print("\nModellspezifikation:")
    print(formula.strip())
    print("\n" + "=" * 72)
    print(f"H1a: Interaktionsmodell – {label}")
    print("=" * 72)
    print(f"Interaktionskoeffizient: {stats['coefficient']:.3f}")
    print(f"Clusterrobuster SE:      {stats['cluster_robust_se']:.3f}")
    print(f"t-Wert:                  {stats['t_value']:.3f}")
    print(f"p-Wert:                  {stats['p_value']:.3f}")
    print(
        f"95%-KI:                  "
        f"[{stats['ci_95_low']:.3f}, {stats['ci_95_high']:.3f}]"
    )
    print(f"H1a unterstützt:         {'Ja' if supported else 'Nein'}")
    print("\nInterpretation:")
    print(interpretation)

    run_wild_cluster_bootstrap(
        model,
        cluster_array,
        "11",
        output_bootstrap_11,
        f"H1a: WCR11 – {label}",
    )
    run_wild_cluster_bootstrap(
        model,
        cluster_array,
        "31",
        output_bootstrap_31,
        f"H1a: WCR31 – {label}",
    )

    simple_slopes = calculate_simple_slopes(
        analysis_df,
        ols_result,
        cluster_result,
        proximity_mean,
    )

    print("\n" + "=" * 72)
    print(f"Simple Slopes – {label}")
    print("=" * 72)
    print(simple_slopes.round(DECIMALS).to_string(index=False))

    predictions = create_predictions(
        analysis_df,
        ols_result,
        afd_support_mean,
        proximity_mean,
        prediction_column,
    )

    create_interaction_plot(
        predictions,
        prediction_column,
        model_spec["y_axis_label"],
        model_spec["figure_title"],
        output_figure,
    )

    model_output = pd.DataFrame(
        [
            {
                "hypothesis": "H1a",
                "operationalization": model_key,
                "model_label": label,
                "model": "party_and_period_fixed_effects",
                "outcome": outcome,
                "predictor": AFD_SUPPORT,
                "moderator": MODERATOR,
                "interaction": INTERACTION,
                "expected_direction": expected_direction,
                **stats,
                "n_observations": int(ols_result.nobs),
                "n_clusters": int(analysis_df[PARTY].nunique()),
                "n_periods": int(analysis_df[PERIOD].nunique()),
                "r_squared": float(ols_result.rsquared),
                "adjusted_r_squared": float(ols_result.rsquared_adj),
                "afd_support_mean": afd_support_mean,
                "proximity_mean": proximity_mean,
                "supported": supported,
                "interpretation": interpretation,
            }
        ]
    )

    save_dataframe(model_output, output_model)
    save_dataframe(simple_slopes, output_simple_slopes)
    save_dataframe(predictions, output_predictions)
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
        f"Datei nicht gefunden:\n{INPUT_FILE.resolve()}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 72)
print("H1a: Moderation durch ideologische Nähe zur AfD")
print("=" * 72)
print(f"Datei: {INPUT_FILE}")
print(f"Zeilen im Gesamtdatensatz: {len(df)}")
print(f"Spalten im Gesamtdatensatz: {len(df.columns)}")


# ============================================================
# 4. BEIDE H1a-OPERATIONALISIERUNGEN SCHÄTZEN
# ============================================================

all_main_results = []
all_simple_slopes = []

for model_key, model_spec in MODEL_SPECS.items():
    results = run_model_specification(
        data=df,
        model_key=model_key,
        model_spec=model_spec,
    )

    all_main_results.append(results["main"])

    slopes = results["simple_slopes"].copy()
    slopes.insert(0, "operationalization", model_key)
    slopes.insert(1, "model_label", model_spec["label"])
    all_simple_slopes.append(slopes)


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

OUTPUT_COMBINED_MODELS = OUTPUT_DIR / "h1a_combined_models.csv"
OUTPUT_COMBINED_SIMPLE_SLOPES = (
    OUTPUT_DIR / "h1a_combined_simple_slopes.csv"
)

save_dataframe(combined_main_results, OUTPUT_COMBINED_MODELS)
save_dataframe(combined_simple_slopes, OUTPUT_COMBINED_SIMPLE_SLOPES)


# ============================================================
# 6. ABSCHLUSSAUSGABE
# ============================================================

print("\n\n" + "=" * 72)
print("H1a: ALLE ANALYSEN ABGESCHLOSSEN")
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
print(f"- Interaktionsmodelle: {OUTPUT_COMBINED_MODELS}")
print(f"- Simple Slopes:       {OUTPUT_COMBINED_SIMPLE_SLOPES}")

print(
    "\nHinweis zur Interpretation:\n"
    "- Beim Distanzmodell unterstützt ein negativer "
    "Interaktionskoeffizient H1a.\n"
    "- Beim Modell der absoluten GAL-TAN-Position unterstützt "
    "ein positiver Interaktionskoeffizient H1a, sofern höhere "
    "Werte den TAN-Pol abbilden.\n"
    "- Der Haupteffekt der zeitkonstanten Nähevariable wird durch "
    "die Partei-Fixed-Effects absorbiert."
)