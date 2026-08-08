# ============================================================
# H1c: MODERATION DURCH DEN OPPOSITIONSSTATUS
#
# is_opposition: 0 = Regierung, 1 = Opposition
#
# MODELL A: gal_tan_distance_to_afd
# Erwartung: afd_support_c × is_opposition < 0
#
# MODELL B: mean_gal_tan_score
# Erwartung: afd_support_c × is_opposition > 0,
# sofern höhere Werte den TAN-Pol abbilden.
#
# Beide Modelle enthalten Partei- und Wahlperioden-FE.
# Alle Ergebniswerte werden mit drei Nachkommastellen ausgegeben.
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from wildboottest.wildboottest import wildboottest

INPUT_FILE = Path(
    "data/processed/final_analysis/"
    "05_party_polling_interval_distances.csv"
)
OUTPUT_DIR = Path("results/hypothesis1c")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AFD_SUPPORT = "afd_support_t"
OPPOSITION = "is_opposition"
PARTY = "party_clean"
PERIOD = "wahlperiode_cat"
MAINSTREAM = "is_mainstream_party"

AFD_SUPPORT_C = "afd_support_c"
INTERACTION = "afd_x_opposition"

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
        "prefix": "h1c_distance",
        "prediction": "predicted_distance",
        "ylabel": "Vorhergesagte GAL–TAN-Distanz zur AfD",
    },
    "absolute": {
        "label": "Absolute GAL-TAN-Position",
        "outcome": "mean_gal_tan_score",
        "expected_direction": "positive",
        "expected_sign": 1,
        "prefix": "h1c_absolute_gal_tan",
        "prediction": "predicted_gal_tan",
        "ylabel": "Vorhergesagte absolute GAL–TAN-Position",
    },
}


def mainstream_mask(series: pd.Series) -> pd.Series:
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


def save_df(dataframe: pd.DataFrame, path: Path) -> None:
    dataframe.to_csv(
        path,
        index=False,
        float_format="%.3f",
    )


def save_bootstrap(result, path: Path) -> None:
    if isinstance(result, pd.DataFrame):
        result.reset_index().to_csv(
            path,
            index=False,
            float_format="%.3f",
        )
    else:
        pd.DataFrame({"result": [str(result)]}).to_csv(
            path,
            index=False,
        )


def direction_matches(coefficient: float, expected_sign: int) -> bool:
    return coefficient < 0 if expected_sign == -1 else coefficient > 0


def prepare_sample(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    required = [outcome, AFD_SUPPORT, OPPOSITION, PARTY, PERIOD, MAINSTREAM]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(
            "Folgende Variablen fehlen:\n"
            + "\n".join(f"- {column}" for column in missing)
        )

    mask = data[required].notna().all(axis=1)
    mask &= mainstream_mask(data[MAINSTREAM])

    sample = data.loc[mask].copy()

    for column in [outcome, AFD_SUPPORT, OPPOSITION]:
        sample[column] = pd.to_numeric(sample[column], errors="coerce")

    sample = sample.dropna(subset=required).copy()

    invalid = ~sample[OPPOSITION].isin([0, 1])
    if invalid.any():
        values = sorted(sample.loc[invalid, OPPOSITION].unique())
        raise ValueError(
            "is_opposition enthält andere Werte als 0 und 1: "
            f"{values}"
        )

    sample[OPPOSITION] = sample[OPPOSITION].astype(int)
    sample[PARTY] = sample[PARTY].astype("category")
    sample[PERIOD] = sample[PERIOD].astype("category")
    sample["cluster_id"] = sample[PARTY].cat.codes.astype(np.int64)

    afd_support_mean = sample[AFD_SUPPORT].mean()
    sample[AFD_SUPPORT_C] = sample[AFD_SUPPORT] - afd_support_mean
    sample[INTERACTION] = sample[AFD_SUPPORT_C] * sample[OPPOSITION]

    return sample


def parameter_stats(ols_result, robust_result, parameter: str) -> dict:
    names = ols_result.model.exog_names
    if parameter not in names:
        raise ValueError(f"Parameter '{parameter}' wurde nicht gefunden.")

    index = names.index(parameter)
    ci = robust_result.conf_int()[index]

    return {
        "coefficient": float(robust_result.params[index]),
        "cluster_robust_se": float(robust_result.bse[index]),
        "t_value": float(robust_result.tvalues[index]),
        "p_value": float(robust_result.pvalues[index]),
        "ci_95_low": float(ci[0]),
        "ci_95_high": float(ci[1]),
    }


def run_bootstrap(model, clusters, bootstrap_type: str, output: Path, label: str):
    print("\n" + "=" * 72)
    print(f"H1c: WCR{bootstrap_type} – {label}")
    print("=" * 72)

    try:
        result = wildboottest(
            model,
            param=INTERACTION,
            cluster=clusters,
            B=BOOTSTRAP_REPLICATIONS,
            bootstrap_type=bootstrap_type,
            impose_null=True,
            seed=SEED,
            show=False,
        )
        print(result.round(DECIMALS) if isinstance(result, pd.DataFrame) else result)
        save_bootstrap(result, output)
        return result
    except Exception as error:
        print(f"Bootstrap fehlgeschlagen: {type(error).__name__}: {error}")
        pd.DataFrame(
            {
                "error_type": [type(error).__name__],
                "error_message": [str(error)],
            }
        ).to_csv(output, index=False)
        return None


def simple_slopes(ols_result, robust_result) -> pd.DataFrame:
    names = ols_result.model.exog_names
    afd_index = names.index(AFD_SUPPORT_C)
    interaction_index = names.index(INTERACTION)
    rows = []

    for label, opposition_value in {"Regierung": 0, "Opposition": 1}.items():
        restriction = np.zeros(len(names))
        restriction[afd_index] = 1
        restriction[interaction_index] = opposition_value

        test = robust_result.t_test(restriction)
        ci = np.asarray(test.conf_int(alpha=ALPHA)).squeeze()
        slope = float(np.asarray(test.effect).squeeze())
        se = float(np.asarray(test.sd).squeeze())
        t_value = float(np.asarray(test.tvalue).squeeze())
        p_value = float(np.asarray(test.pvalue).squeeze())

        rows.append(
            {
                "status": label,
                OPPOSITION: opposition_value,
                "effect_afd_support": slope,
                "cluster_robust_se": se,
                "t_value": t_value,
                "p_value": p_value,
                "ci_95_low": float(ci[0]),
                "ci_95_high": float(ci[1]),
                "significant_05": p_value < ALPHA,
            }
        )

    return pd.DataFrame(rows)


def predictions(
    sample: pd.DataFrame,
    ols_result,
    afd_support_mean: float,
    prediction_column: str,
) -> pd.DataFrame:
    afd_grid = np.linspace(
        sample[AFD_SUPPORT].min(),
        sample[AFD_SUPPORT].max(),
        100,
    )
    reference_party = sample[PARTY].cat.categories[0]
    reference_period = sample[PERIOD].cat.categories[0]
    frames = []

    for label, opposition_value in {"Regierung": 0, "Opposition": 1}.items():
        prediction_data = pd.DataFrame(
            {
                AFD_SUPPORT: afd_grid,
                OPPOSITION: opposition_value,
                PARTY: reference_party,
                PERIOD: reference_period,
            }
        )
        prediction_data[AFD_SUPPORT_C] = (
            prediction_data[AFD_SUPPORT] - afd_support_mean
        )
        prediction_data[INTERACTION] = (
            prediction_data[AFD_SUPPORT_C]
            * prediction_data[OPPOSITION]
        )

        prediction = ols_result.get_prediction(
            prediction_data
        ).summary_frame(alpha=ALPHA)

        prediction_data["status"] = label
        prediction_data[prediction_column] = prediction["mean"]
        prediction_data["ci_95_low"] = prediction["mean_ci_lower"]
        prediction_data["ci_95_high"] = prediction["mean_ci_upper"]
        frames.append(prediction_data)

    return pd.concat(frames, ignore_index=True)


def make_plot(
    prediction_data: pd.DataFrame,
    prediction_column: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    plt.figure(figsize=(9, 6))

    for label, group in prediction_data.groupby("status", sort=False):
        plt.plot(
            group[AFD_SUPPORT],
            group[prediction_column],
            label=label,
        )

    plt.xlabel("AfD-Unterstützung in Prozentpunkten")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def run_model(data: pd.DataFrame, model_key: str, spec: dict) -> dict:
    label = spec["label"]
    outcome = spec["outcome"]
    prefix = spec["prefix"]
    expected_sign = spec["expected_sign"]

    outputs = {
        "model": OUTPUT_DIR / f"{prefix}_model_results.csv",
        "bootstrap_11": OUTPUT_DIR / f"{prefix}_bootstrap_wcr11.csv",
        "bootstrap_31": OUTPUT_DIR / f"{prefix}_bootstrap_wcr31.csv",
        "slopes": OUTPUT_DIR / f"{prefix}_simple_slopes.csv",
        "predictions": OUTPUT_DIR / f"{prefix}_predicted_values.csv",
        "figure": OUTPUT_DIR / f"{prefix}_interaction_plot.png",
        "sample": OUTPUT_DIR / f"{prefix}_analysis_sample.csv",
    }

    sample = prepare_sample(data, outcome)
    afd_support_mean = float(sample[AFD_SUPPORT].mean())
    clusters = sample["cluster_id"].to_numpy(dtype=np.int64)

    print("\n\n" + "#" * 72)
    print(f"STARTE H1c-MODELL: {label.upper()}")
    print("#" * 72)
    print(f"Analyseeinheiten: {len(sample)}")
    print(f"Parteien/Cluster: {sample[PARTY].nunique()}")
    print(f"Wahlperioden: {sample[PERIOD].nunique()}")

    print("\nBeobachtungen nach Status:")
    print(
        sample[OPPOSITION]
        .value_counts()
        .sort_index()
        .rename(index={0: "Regierung", 1: "Opposition"})
    )

    print("\nStatus nach Partei:")
    print(
        pd.crosstab(sample[PARTY], sample[OPPOSITION]).rename(
            columns={0: "Regierung", 1: "Opposition"}
        )
    )

    print("\nMittelwert für die Zentrierung:")
    print(f"AfD-Unterstützung: {afd_support_mean:.3f}")

    formula = f"""
        {outcome}
        ~ {AFD_SUPPORT_C}
        + {OPPOSITION}
        + {INTERACTION}
        + C({PARTY})
        + C({PERIOD})
    """

    model = smf.ols(formula=formula, data=sample)
    ols_result = model.fit()

    robust_result = ols_result.get_robustcov_results(
        cov_type="cluster",
        groups=clusters,
        use_correction=True,
        df_correction=True,
        use_t=True,
    )

    stats = parameter_stats(
        ols_result,
        robust_result,
        INTERACTION,
    )

    supported = (
        direction_matches(stats["coefficient"], expected_sign)
        and stats["p_value"] < ALPHA
    )

    print("\nModellspezifikation:")
    print(formula.strip())
    print("\n" + "=" * 72)
    print(f"H1c: Interaktionsmodell – {label}")
    print("=" * 72)
    print(f"Interaktionskoeffizient: {stats['coefficient']:.3f}")
    print(f"Clusterrobuster SE:      {stats['cluster_robust_se']:.3f}")
    print(f"t-Wert:                  {stats['t_value']:.3f}")
    print(f"p-Wert:                  {stats['p_value']:.3f}")
    print(
        "95%-KI:                  "
        f"[{stats['ci_95_low']:.3f}, {stats['ci_95_high']:.3f}]"
    )
    print(f"H1c unterstützt:         {'Ja' if supported else 'Nein'}")

    run_bootstrap(
        model,
        clusters,
        "11",
        outputs["bootstrap_11"],
        label,
    )
    run_bootstrap(
        model,
        clusters,
        "31",
        outputs["bootstrap_31"],
        label,
    )

    slopes = simple_slopes(ols_result, robust_result)
    print("\n" + "=" * 72)
    print(f"Simple Slopes – {label}")
    print("=" * 72)
    print(slopes.round(DECIMALS).to_string(index=False))

    prediction_data = predictions(
        sample,
        ols_result,
        afd_support_mean,
        spec["prediction"],
    )

    make_plot(
        prediction_data,
        spec["prediction"],
        spec["ylabel"],
        f"H1c: Oppositionsstatus – {label}",
        outputs["figure"],
    )

    interpretation = (
        f"{label}: Der Interaktionseffekt weist in die erwartete "
        "Richtung und ist statistisch signifikant."
        if supported
        else f"{label}: H1c wird durch dieses Modell nicht unterstützt."
    )

    model_output = pd.DataFrame(
        [
            {
                "hypothesis": "H1c",
                "operationalization": model_key,
                "model_label": label,
                "model": "party_and_period_fixed_effects",
                "outcome": outcome,
                "predictor": AFD_SUPPORT,
                "moderator": OPPOSITION,
                "interaction": INTERACTION,
                "expected_direction": spec["expected_direction"],
                **stats,
                "n_observations": int(ols_result.nobs),
                "n_clusters": int(sample[PARTY].nunique()),
                "n_periods": int(sample[PERIOD].nunique()),
                "r_squared": float(ols_result.rsquared),
                "adjusted_r_squared": float(ols_result.rsquared_adj),
                "afd_support_mean": afd_support_mean,
                "supported": supported,
                "interpretation": interpretation,
            }
        ]
    )

    save_df(model_output, outputs["model"])
    save_df(slopes, outputs["slopes"])
    save_df(prediction_data, outputs["predictions"])
    sample.to_csv(
        outputs["sample"],
        index=False,
        float_format="%.3f",
    )

    print("\nGespeicherte Dateien:")
    for output in outputs.values():
        print(f"- {output}")

    slopes = slopes.copy()
    slopes.insert(0, "operationalization", model_key)
    slopes.insert(1, "model_label", label)

    return {
        "main": model_output,
        "slopes": slopes,
    }


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Datei nicht gefunden:\n{INPUT_FILE.resolve()}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 72)
print("H1c: Moderation durch den Oppositionsstatus")
print("=" * 72)
print(f"Datei: {INPUT_FILE}")
print(f"Zeilen im Gesamtdatensatz: {len(df)}")
print(f"Spalten im Gesamtdatensatz: {len(df.columns)}")

main_results = []
slope_results = []

for model_key, model_spec in MODEL_SPECS.items():
    result = run_model(df, model_key, model_spec)
    main_results.append(result["main"])
    slope_results.append(result["slopes"])

combined_main = pd.concat(main_results, ignore_index=True)
combined_slopes = pd.concat(slope_results, ignore_index=True)

OUTPUT_COMBINED_MODELS = OUTPUT_DIR / "h1c_combined_models.csv"
OUTPUT_COMBINED_SLOPES = OUTPUT_DIR / "h1c_combined_simple_slopes.csv"

save_df(combined_main, OUTPUT_COMBINED_MODELS)
save_df(combined_slopes, OUTPUT_COMBINED_SLOPES)

print("\n\n" + "=" * 72)
print("H1c: ALLE ANALYSEN ABGESCHLOSSEN")
print("=" * 72)
print("\nInteraktionsmodelle im Vergleich:")
print(
    combined_main[
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
print(f"- Simple Slopes:       {OUTPUT_COMBINED_SLOPES}")

print(
    "\nHinweis zur Interpretation:\n"
    "- Beim Distanzmodell unterstützt ein negativer "
    "Interaktionskoeffizient H1c.\n"
    "- Beim Modell der absoluten GAL-TAN-Position unterstützt "
    "ein positiver Interaktionskoeffizient H1c, sofern höhere "
    "Werte den TAN-Pol abbilden.\n"
    "- is_opposition = 0 bezeichnet Regierungsparteien und "
    "is_opposition = 1 Oppositionsparteien."
)