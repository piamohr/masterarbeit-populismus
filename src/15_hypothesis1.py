# =====================================================
# H1: POSITIONELLE ANNÄHERUNG UND ABSOLUTE GAL-TAN-POSITION
#
# Dieses Skript prüft H1 auf zwei Arten:
#
# MODELL A: Positionelle Annäherung an die AfD
#   AV: gal_tan_distance_to_afd
#   Erwartung: negativer Koeffizient
#   Interpretation:
#   Mit steigender AfD-Unterstützung wird die absolute
#   GAL-TAN-Distanz zwischen Mainstream-Partei und AfD kleiner.
#
# MODELL B: Absolute GAL-TAN-Position der Mainstream-Parteien
#   AV: mean_gal_tan_score
#   Erwartung: positiver Koeffizient
#   Interpretation:
#   Mit steigender AfD-Unterstützung positionieren sich
#   Mainstream-Parteien stärker in Richtung des TAN-Pols.
#
# Voraussetzung für Modell B:
#   Höhere Werte der GAL-TAN-Skala stehen für stärker
#   traditionalistische, autoritäre und nationalistische Positionen.
#
# Für beide Modelle:
#   - Partei-Fixed-Effects
#   - Wahlperioden-Fixed-Effects
#   - Clusterrobuste Standardfehler auf Parteiebene
#   - Small-Sample-Korrektur
#   - Wild-Cluster-Bootstrap WCR11 und WCR31
#
# Alle Ergebniswerte werden mit drei Nachkommastellen
# ausgegeben und gespeichert.
# =====================================================

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from wildboottest.wildboottest import wildboottest


# =====================================================
# 1. EINSTELLUNGEN
# =====================================================

INPUT_FILE = Path(
    "data/processed/final_analysis/05_party_polling_interval_distances.csv"
)

OUTPUT_DIR = Path("results/hypothesis1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTOR = "afd_support_t"
PARTY = "party_clean"
PERIOD = "wahlperiode_cat"
MAINSTREAM = "is_mainstream_party"
POLL_INTERVAL = "poll_interval_id"

BOOTSTRAP_REPLICATIONS = 9999
SEED = 12345
DECIMALS = 3

# Beide H1-Operationalisierungen.
MODEL_SPECS = {
    "distance": {
        "label": "Positionelle Annäherung",
        "outcome": "gal_tan_distance_to_afd",
        "expected_direction": "negative",
        "expected_sign": -1,
        "output_prefix": "h1_distance",
        "descriptive_variables": [
            "gal_tan_distance_to_afd",
            "afd_support_t",
            "mean_gal_tan_score",
            "afd_mean_gal_tan_score",
            "n_speeches_with_gal_tan",
            "afd_n_speeches_with_gal_tan",
        ],
    },
    "absolute": {
        "label": "Absolute GAL-TAN-Position",
        "outcome": "mean_gal_tan_score",
        "expected_direction": "positive",
        "expected_sign": 1,
        "output_prefix": "h1_absolute_gal_tan",
        "descriptive_variables": [
            "mean_gal_tan_score",
            "afd_support_t",
            "afd_mean_gal_tan_score",
            "gal_tan_distance_to_afd",
            "n_speeches_with_gal_tan",
            "afd_n_speeches_with_gal_tan",
        ],
    },
}


# =====================================================
# 2. HILFSFUNKTIONEN
# =====================================================

def is_true_like(series):
    """
    Wandelt verschiedene mögliche Darstellungen einer booleschen
    Variable in eine boolesche Maske um.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "ja"])
    )


def save_bootstrap_result(result, output_path):
    """
    Speichert das Ergebnis von wildboottest als CSV.
    """
    if isinstance(result, pd.DataFrame):
        result.reset_index().to_csv(
            output_path,
            index=False,
            float_format="%.3f",
        )
    else:
        pd.DataFrame(
            {"result": [str(result)]}
        ).to_csv(output_path, index=False)


def save_bootstrap_error(error, output_path):
    """
    Speichert Fehlertyp und Fehlermeldung eines Bootstrap-Laufs.
    """
    pd.DataFrame(
        {
            "error_type": [type(error).__name__],
            "error_message": [str(error)],
        }
    ).to_csv(output_path, index=False)


def extract_cluster_result(
    fitted_ols,
    fitted_cluster,
    parameter,
):
    """
    Extrahiert Koeffizient, clusterrobusten Standardfehler,
    t-Wert, p-Wert und 95%-Konfidenzintervall.
    """
    parameter_names = fitted_ols.model.exog_names

    if parameter not in parameter_names:
        raise ValueError(
            f"Der Parameter '{parameter}' wurde nicht "
            "in der Modellmatrix gefunden."
        )

    parameter_index = parameter_names.index(parameter)
    confidence_interval = fitted_cluster.conf_int()[parameter_index]

    return {
        "coefficient": float(
            fitted_cluster.params[parameter_index]
        ),
        "cluster_robust_se": float(
            fitted_cluster.bse[parameter_index]
        ),
        "t_value": float(
            fitted_cluster.tvalues[parameter_index]
        ),
        "p_value": float(
            fitted_cluster.pvalues[parameter_index]
        ),
        "ci_95_low": float(confidence_interval[0]),
        "ci_95_high": float(confidence_interval[1]),
    }


def interpret_result(
    coefficient,
    p_value,
    expected_sign,
    model_label,
):
    """
    Erstellt eine automatische, richtungsbezogene Interpretation.
    """
    direction_matches = (
        coefficient < 0
        if expected_sign == -1
        else coefficient > 0
    )

    if direction_matches and p_value < 0.05:
        return (
            f"{model_label}: Der Koeffizient weist in die erwartete "
            "Richtung und ist statistisch signifikant."
        )

    if direction_matches:
        return (
            f"{model_label}: Der Koeffizient weist in die erwartete "
            "Richtung, ist aber nicht statistisch signifikant."
        )

    if p_value < 0.05:
        return (
            f"{model_label}: Der Koeffizient ist statistisch signifikant, "
            "weist aber in die der Hypothese entgegengesetzte Richtung."
        )

    return (
        f"{model_label}: Der Koeffizient weist nicht in die erwartete "
        "Richtung und ist nicht statistisch signifikant."
    )


def run_wild_cluster_bootstrap(
    model,
    parameter,
    cluster_array,
    bootstrap_type,
    output_path,
    heading,
):
    """
    Führt einen Wild-Cluster-Bootstrap aus und speichert das Ergebnis.
    """
    print("\n" + "=" * 70)
    print(heading)
    print("=" * 70)

    try:
        result = wildboottest(
            model,
            param=parameter,
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

        save_bootstrap_result(result, output_path)
        return result

    except Exception as error:
        print(
            f"Der Wild-Cluster-Bootstrap {bootstrap_type} "
            "konnte nicht berechnet werden."
        )
        print(f"Fehlertyp: {type(error).__name__}")
        print(f"Fehlermeldung: {error}")

        save_bootstrap_error(error, output_path)
        return None


def prepare_analysis_sample(
    data,
    outcome,
):
    """
    Erstellt für eine abhängige Variable die vollständige
    Mainstream-Analysestichprobe.
    """
    required_columns = list(
        dict.fromkeys(
            [
                outcome,
                PREDICTOR,
                PARTY,
                PERIOD,
                MAINSTREAM,
                POLL_INTERVAL,
            ]
            + MODEL_SPECS[
                next(
                    key
                    for key, spec in MODEL_SPECS.items()
                    if spec["outcome"] == outcome
                )
            ]["descriptive_variables"]
        )
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Folgende benötigte Variablen fehlen im Datensatz:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_columns
            )
        )

    analysis_filter = (
        data[outcome].notna()
        & data[PREDICTOR].notna()
        & data[PARTY].notna()
        & data[PERIOD].notna()
        & data[MAINSTREAM].notna()
        & is_true_like(data[MAINSTREAM])
    )

    analysis_df = data.loc[analysis_filter].copy()

    if analysis_df.empty:
        raise ValueError(
            f"Nach Anwendung der Filter enthält die Stichprobe "
            f"für '{outcome}' keine Beobachtungen."
        )

    analysis_df[PARTY] = analysis_df[PARTY].astype("category")
    analysis_df[PERIOD] = analysis_df[PERIOD].astype("category")

    analysis_df["cluster_id"] = (
        analysis_df[PARTY]
        .cat.codes
        .astype(np.int64)
    )

    if (analysis_df["cluster_id"] < 0).any():
        raise ValueError(
            "Bei der Erstellung der numerischen Cluster-ID "
            "sind ungültige Werte entstanden."
        )

    return analysis_df


def check_panel_structure(
    analysis_df,
    model_label,
):
    """
    Gibt zentrale Informationen zur Panelstruktur aus und prüft
    doppelte Partei-Intervall-Kombinationen.
    """
    print("\n" + "=" * 70)
    print(f"H1: Analysestichprobe – {model_label}")
    print("=" * 70)

    print(f"Analyseeinheiten: {len(analysis_df)}")
    print(
        "Zahl der Parteien/Cluster: "
        f"{analysis_df[PARTY].nunique()}"
    )
    print(
        "Zahl der Wahlperioden: "
        f"{analysis_df[PERIOD].nunique()}"
    )

    print(
        "Zahl der Umfrageintervalle: "
        f"{analysis_df[POLL_INTERVAL].nunique()}"
    )

    print("\nBeobachtungen je Partei:")
    print(
        analysis_df.groupby(
            PARTY,
            observed=True,
        ).size()
    )

    print("\nBeobachtungen je Wahlperiode:")
    print(
        analysis_df.groupby(
            PERIOD,
            observed=True,
        ).size()
    )

    print("\nZuordnung der numerischen Cluster-IDs:")
    print(
        analysis_df[
            [PARTY, "cluster_id"]
        ]
        .drop_duplicates()
        .sort_values("cluster_id")
        .to_string(index=False)
    )

    duplicate_rows = analysis_df.duplicated(
        subset=[PARTY, POLL_INTERVAL],
        keep=False,
    )

    n_duplicate_rows = int(duplicate_rows.sum())

    print(
        "\nDoppelte Partei-Intervall-Zeilen: "
        f"{n_duplicate_rows}"
    )

    if n_duplicate_rows > 0:
        duplicate_examples = (
            analysis_df.loc[
                duplicate_rows,
                [PARTY, POLL_INTERVAL],
            ]
            .sort_values([PARTY, POLL_INTERVAL])
        )

        print("\nBeispiele doppelter Kombinationen:")
        print(
            duplicate_examples
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            "Die Analysedatei enthält doppelte Kombinationen "
            "aus Partei und Umfrageintervall."
        )


def fit_cluster_model(
    analysis_df,
    formula,
):
    """
    Schätzt ein OLS-Modell mit Dummy-Variablen für die Fixed Effects
    und berechnet clusterrobuste Inferenz auf Parteiebene.
    """
    model = smf.ols(
        formula=formula,
        data=analysis_df,
    )

    fitted_ols = model.fit()

    cluster_array = analysis_df["cluster_id"].to_numpy(
        dtype=np.int64
    )

    n_model_rows = model.exog.shape[0]
    n_cluster_rows = len(cluster_array)

    print("\nZeilen im Analysedatensatz:", len(analysis_df))
    print("Zeilen in der Modellmatrix:", n_model_rows)
    print("Zeilen im Cluster-Array:", n_cluster_rows)

    if not (
        len(analysis_df)
        == n_model_rows
        == n_cluster_rows
    ):
        raise ValueError(
            "Die Zahl der Modellzeilen stimmt nicht mit der Zahl "
            "der Clusterkennungen überein."
        )

    fitted_cluster = (
        fitted_ols.get_robustcov_results(
            cov_type="cluster",
            groups=cluster_array,
            use_correction=True,
            df_correction=True,
            use_t=True,
        )
    )

    return (
        model,
        fitted_ols,
        fitted_cluster,
        cluster_array,
    )


def run_model_specification(
    data,
    model_key,
    model_spec,
):
    """
    Führt Hauptmodell und Bootstrap-Tests für eine
    H1-Operationalisierung vollständig aus.
    """
    label = model_spec["label"]
    outcome = model_spec["outcome"]
    expected_direction = model_spec["expected_direction"]
    expected_sign = model_spec["expected_sign"]
    prefix = model_spec["output_prefix"]

    output_analysis_sample = (
        OUTPUT_DIR / f"{prefix}_analysis_sample.csv"
    )
    output_model_results = (
        OUTPUT_DIR / f"{prefix}_model_results.csv"
    )
    output_bootstrap_11 = (
        OUTPUT_DIR / f"{prefix}_wild_cluster_bootstrap_11.csv"
    )
    output_bootstrap_31 = (
        OUTPUT_DIR / f"{prefix}_wild_cluster_bootstrap_31.csv"
    )

    analysis_df = prepare_analysis_sample(
        data=data,
        outcome=outcome,
    )

    check_panel_structure(
        analysis_df=analysis_df,
        model_label=label,
    )

    print("\n" + "=" * 70)
    print(f"H1: Deskriptive Werte – {label}")
    print("=" * 70)

    descriptive_variables = model_spec["descriptive_variables"]

    print(
        analysis_df[descriptive_variables]
        .describe()
        .round(DECIMALS)
    )

    # -------------------------------------------------
    # Hauptmodell mit Partei- und Wahlperioden-FE
    # -------------------------------------------------

    formula_main = f"""
        {outcome}
        ~ {PREDICTOR}
        + C({PARTY})
        + C({PERIOD})
    """

    print("\n" + "=" * 70)
    print(f"H1: Hauptmodell – {label}")
    print("=" * 70)
    print(formula_main.strip())

    (
        model_main,
        fitted_main_ols,
        fitted_main_cluster,
        cluster_array,
    ) = fit_cluster_model(
        analysis_df=analysis_df,
        formula=formula_main,
    )

    estimates_main = extract_cluster_result(
        fitted_ols=fitted_main_ols,
        fitted_cluster=fitted_main_cluster,
        parameter=PREDICTOR,
    )

    interpretation_main = interpret_result(
        coefficient=estimates_main["coefficient"],
        p_value=estimates_main["p_value"],
        expected_sign=expected_sign,
        model_label=label,
    )

    n_observations = int(fitted_main_ols.nobs)
    n_clusters = int(analysis_df[PARTY].nunique())

    print(f"Beobachtungen:          {n_observations}")
    print(f"Parteien/Cluster:       {n_clusters}")
    print(
        "Koeffizient:            "
        f"{estimates_main['coefficient']:.3f}"
    )
    print(
        "Clusterrobuster SE:     "
        f"{estimates_main['cluster_robust_se']:.3f}"
    )
    print(
        "t-Wert:                 "
        f"{estimates_main['t_value']:.3f}"
    )
    print(
        "p-Wert:                 "
        f"{estimates_main['p_value']:.3f}"
    )
    print(
        "95%-Konfidenzintervall: "
        f"[{estimates_main['ci_95_low']:.3f}, "
        f"{estimates_main['ci_95_high']:.3f}]"
    )

    print("\nInterpretation:")
    print(interpretation_main)

    main_result = pd.DataFrame(
        [
            {
                "hypothesis": "H1",
                "operationalization": model_key,
                "model_label": label,
                "model": "party_and_period_fixed_effects",
                "outcome": outcome,
                "predictor": PREDICTOR,
                "expected_direction": expected_direction,
                **estimates_main,
                "n_observations": n_observations,
                "n_clusters": n_clusters,
                "r_squared": float(fitted_main_ols.rsquared),
                "adjusted_r_squared": float(
                    fitted_main_ols.rsquared_adj
                ),
                "interpretation": interpretation_main,
            }
        ]
    )

    main_result.to_csv(
        output_model_results,
        index=False,
        float_format="%.3f",
    )

    analysis_df.to_csv(
        output_analysis_sample,
        index=False,
        float_format="%.3f",
    )

    run_wild_cluster_bootstrap(
        model=model_main,
        parameter=PREDICTOR,
        cluster_array=cluster_array,
        bootstrap_type="11",
        output_path=output_bootstrap_11,
        heading=f"H1: WCR11 – {label}",
    )

    run_wild_cluster_bootstrap(
        model=model_main,
        parameter=PREDICTOR,
        cluster_array=cluster_array,
        bootstrap_type="31",
        output_path=output_bootstrap_31,
        heading=f"H1: WCR31 – {label}",
    )

    print("\nGespeicherte Dateien für dieses Modell:")
    print(f"- Analysestichprobe: {output_analysis_sample}")
    print(f"- Hauptmodell:       {output_model_results}")
    print(f"- Bootstrap WCR11:   {output_bootstrap_11}")
    print(f"- Bootstrap WCR31:   {output_bootstrap_31}")

    return main_result


# =====================================================
# 3. DATEN EINLESEN
# =====================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Die Eingabedatei wurde nicht gefunden:\n"
        f"{INPUT_FILE.resolve()}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("H1: Daten eingelesen")
print("=" * 70)
print(f"Datei: {INPUT_FILE}")
print(f"Zeilen im Gesamtdatensatz: {len(df)}")
print(f"Spalten im Gesamtdatensatz: {len(df.columns)}")


# =====================================================
# 4. BEIDE H1-OPERATIONALISIERUNGEN SCHÄTZEN
# =====================================================

all_main_results = []

for model_key, model_spec in MODEL_SPECS.items():
    print("\n\n" + "#" * 70)
    print(
        f"STARTE H1-MODELL: "
        f"{model_spec['label'].upper()}"
    )
    print("#" * 70)

    model_results = run_model_specification(
        data=df,
        model_key=model_key,
        model_spec=model_spec,
    )

    all_main_results.append(model_results)


# =====================================================
# 5. GEMEINSAME ERGEBNISTABELLE SPEICHERN
# =====================================================

combined_main_results = pd.concat(
    all_main_results,
    ignore_index=True,
)

output_combined_main = (
    OUTPUT_DIR / "h1_combined_main_models.csv"
)

combined_main_results.to_csv(
    output_combined_main,
    index=False,
    float_format="%.3f",
)


# =====================================================
# 6. ABSCHLUSSAUSGABE
# =====================================================

print("\n\n" + "=" * 70)
print("H1: ALLE ANALYSEN ABGESCHLOSSEN")
print("=" * 70)

print("\nHauptmodelle im Vergleich:")
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
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)

print("\nGemeinsame Ergebnisdatei:")
print(f"- Hauptmodelle: {output_combined_main}")

print(
    "\nHinweis zur Interpretation:\n"
    "- Beim Distanzmodell unterstützt ein negativer Koeffizient H1.\n"
    "- Beim Modell der absoluten GAL-TAN-Position unterstützt ein "
    "positiver Koeffizient H1, sofern höhere Skalenwerte den TAN-Pol "
    "abbilden.\n"
    "- Beide Modelle beantworten unterschiedliche, aber komplementäre "
    "Fragen."
)