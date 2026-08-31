# ============================================================
# FF3 / FF4: Wechselseitige Beziehung zwischen rhetorischer
# und positioneller Entwicklung auf aggregierter Mainstream-Ebene
#
# FF3: Beeinflusst die rhetorische Entwicklung des Mainstreams
#      die spätere positionelle Entwicklung?
#
# FF4: Beeinflusst die positionelle Entwicklung des Mainstreams
#      die spätere rhetorische Entwicklung?
#
# Analyseebene:
# - eine aggregierte Mainstream-Beobachtung je Umfrageintervall
#
# Es werden zwei Operationalisierungen analysiert:
#
# A) DISTANZMODELLE
#    - populism_distance_to_afd
#    - gal_tan_distance_to_afd
#
# B) MODELLE MIT ABSOLUTEN MAINSTREAM-WERTEN
#    - mean_populism_score
#    - mean_gal_tan_score
#
# Verfahren:
# - dynamische lineare Zeitreihenregressionen
# - Wahlperioden-Fixed-Effects
# - HAC-/Newey-West-Standardfehler mit maxlags = 1
# - Kontrolle der verzögerten abhängigen Variable
# - Kontrolle der verzögerten AfD-Unterstützung
#
# Alle tabellarischen Ausgaben verwenden drei Nachkommastellen.
# ============================================================

from pathlib import Path
import warnings

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. PFADE UND VARIABLEN
# ============================================================

INPUT_FILE = Path(
    "data/processed/final_analysis/06_mainstream_polling_interval_distances.csv"
)

OUTPUT_DIR = Path(
    "results/research_question3_4_crosslagged"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Struktur- und Kontrollvariablen
INTERVAL_ID = "poll_interval_id"
TIME_ORDER = "poll_date"
NEXT_POLL_DATE = "next_poll_date"
PERIOD = "wahlperiode"
AFD_SUPPORT = "afd_support_t"

# Distanzwerte zur AfD
POSITION_DISTANCE = "gal_tan_distance_to_afd"
RHETORIC_DISTANCE = "populism_distance_to_afd"

# Absolute Werte des aggregierten Mainstreams
POSITION_ABSOLUTE = "mean_gal_tan_score"
RHETORIC_ABSOLUTE = "mean_populism_score"

# HAC-/Newey-West-Einstellung
HAC_MAXLAGS = 1


# ============================================================
# 2. HILFSFUNKTIONEN
# ============================================================

def make_numeric(series: pd.Series) -> pd.Series:
    """
    Konvertiert numerische Werte robust. Unterstützt auch
    deutsche Dezimaltrennzeichen und Prozentzeichen.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def check_required_columns(
    data: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """Prüft, ob alle für die Analyse benötigten Variablen vorhanden sind."""
    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        available = "\n".join(
            f"  - {column}" for column in data.columns
        )
        missing_text = "\n".join(
            f"  - {column}" for column in missing
        )

        raise KeyError(
            "\nFolgende benötigte Variablen fehlen:\n"
            f"{missing_text}\n\n"
            "Vorhandene Spalten:\n"
            f"{available}"
        )


def create_lag(
    data: pd.DataFrame,
    source: str,
    target: str,
) -> None:
    """
    Erstellt einen Lag erster Ordnung.

    Da der Datensatz genau eine aggregierte Mainstream-Zeitreihe
    enthält, erfolgt keine Gruppierung nach Partei.
    """
    data[target] = data[source].shift(1)


def fit_time_series_model(
    data: pd.DataFrame,
    outcome: str,
    predictor_lag: str,
    outcome_lag: str,
    afd_support_lag: str,
    model_name: str,
):
    """
    Schätzt ein dynamisches Modell auf aggregierter Mainstream-Ebene:

    Y_t = b0
          + b1 * X_(t-1)
          + b2 * Y_(t-1)
          + b3 * AfD-Unterstützung_(t-1)
          + Wahlperioden-FE
          + Fehler_t

    Die Inferenz basiert auf HAC-/Newey-West-Standardfehlern.
    """

    formula = (
        f"{outcome} ~ "
        f"{predictor_lag} + "
        f"{outcome_lag} + "
        f"{afd_support_lag} + "
        f"C({PERIOD})"
    )

    model_columns = [
        outcome,
        predictor_lag,
        outcome_lag,
        afd_support_lag,
        PERIOD,
    ]

    model_df = (
        data[model_columns]
        .dropna()
        .copy()
    )

    # Patsy/statsmodels kann den pandas-eigenen nullable Datentyp
    # Int64Dtype nicht zuverlässig als kategoriale Variable verarbeiten.
    # Nach dem Entfernen fehlender Werte wird die Wahlperiode deshalb
    # in gewöhnliche Python-Strings umgewandelt. C(wahlperiode) behandelt
    # sie anschließend eindeutig als kategoriale Fixed-Effects-Variable.
    model_df[PERIOD] = (
        model_df[PERIOD]
        .astype(int)
        .astype(str)
    )

    if model_df.empty:
        raise ValueError(
            f"{model_name}: Nach Listwise Deletion bleiben "
            "keine Beobachtungen übrig."
        )

    if len(model_df) <= len(model_df.columns):
        raise ValueError(
            f"{model_name}: Es liegen zu wenige vollständige "
            "Beobachtungen für das Modell vor."
        )

    result = smf.ols(
        formula=formula,
        data=model_df,
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_MAXLAGS},
    )

    return result, model_df, formula


def tidy_coefficient_table(
    result,
    focal_term: str,
    model_name: str,
    research_question: str,
    operationalization: str,
) -> pd.DataFrame:
    """Erzeugt eine kompakte Koeffiziententabelle."""
    conf = result.conf_int()

    table = pd.DataFrame({
        "research_question": research_question,
        "operationalization": operationalization,
        "model": model_name,
        "term": result.params.index,
        "focal_term": result.params.index == focal_term,
        "b": result.params.values,
        "SE": result.bse.values,
        "t": result.tvalues.values,
        "p": result.pvalues.values,
        "CI_low": conf.iloc[:, 0].values,
        "CI_high": conf.iloc[:, 1].values,
    })

    return table


def extract_focal_test(
    result,
    focal_term: str,
    research_question: str,
    operationalization: str,
    effect: str,
    n_obs: int,
) -> dict:
    """Extrahiert den zentralen zeitverzögerten Effekt."""
    ci_low, ci_high = result.conf_int().loc[focal_term]

    return {
        "research_question": research_question,
        "operationalization": operationalization,
        "effect": effect,
        "focal_term": focal_term,
        "N": int(n_obs),
        "b": result.params[focal_term],
        "SE": result.bse[focal_term],
        "t": result.tvalues[focal_term],
        "p_raw": result.pvalues[focal_term],
        "CI_low": ci_low,
        "CI_high": ci_high,
        "R2": result.rsquared,
        "R2_adj": result.rsquared_adj,
    }


def format_model_report(
    title: str,
    result,
    model_df: pd.DataFrame,
    formula: str,
    focal_term: str,
    interpretation_type: str,
) -> str:
    """Erstellt einen ausführlichen Bericht für ein einzelnes Modell."""
    b = result.params[focal_term]
    se = result.bse[focal_term]
    t_value = result.tvalues[focal_term]
    p_value = result.pvalues[focal_term]
    ci_low, ci_high = result.conf_int().loc[focal_term]

    lines = [
        "=" * 88,
        title,
        "=" * 88,
        f"Formel: {formula}",
        f"N Beobachtungen: {int(result.nobs)}",
        f"Anzahl Wahlperioden: {model_df[PERIOD].nunique()}",
        f"HAC-/Newey-West-Standardfehler: maxlags = {HAC_MAXLAGS}",
        f"R²: {result.rsquared:.3f}",
        f"Adjustiertes R²: {result.rsquared_adj:.3f}",
        "",
        "Zentraler zeitverzögerter Effekt:",
        (
            f"b = {b:.3f}, SE = {se:.3f}, t = {t_value:.3f}, "
            f"p = {p_value:.3f}, "
            f"95%-KI [{ci_low:.3f}, {ci_high:.3f}]"
        ),
        "",
    ]

    if interpretation_type == "distance":
        lines.extend([
            "Hinweis zur Interpretation der Distanzmodelle:",
            (
                "- Ein positiver Koeffizient bedeutet, dass größere Distanzen "
                "in t-1 mit größeren Distanzen in t zusammenhängen. Umgekehrt "
                "geht damit stärkere Annäherung in t-1 mit stärkerer "
                "Annäherung in t einher."
            ),
            (
                "- Ein negativer Koeffizient bedeutet, dass größere Distanzen "
                "in t-1 mit kleineren Distanzen in t zusammenhängen. Die beiden "
                "Annäherungsformen entwickeln sich dann gegenläufig."
            ),
            "",
        ])

    elif interpretation_type == "absolute":
        lines.extend([
            "Hinweis zur Interpretation der absoluten Werte:",
            (
                "- Ein positiver Koeffizient bedeutet, dass ein höherer Wert "
                "der erklärenden Dimension in t-1 mit einem höheren Wert der "
                "abhängigen Dimension in t zusammenhängt."
            ),
            (
                "- Ein negativer Koeffizient bedeutet, dass ein höherer Wert "
                "der erklärenden Dimension in t-1 mit einem niedrigeren Wert "
                "der abhängigen Dimension in t zusammenhängt."
            ),
            (
                "- Die inhaltliche Richtung des GAL–TAN-Scores richtet sich "
                "nach der Kodierung des Ausgangsdatensatzes. Vor der "
                "substanziellen Interpretation ist daher zu prüfen, ob höhere "
                "Werte den TAN- oder den GAL-Pol kennzeichnen."
            ),
            "",
        ])

    lines.extend([
        "Vollständige Modellzusammenfassung:",
        result.summary().as_text(),
        "",
    ])

    return "\n".join(lines)


# ============================================================
# 3. DATEN EINLESEN UND AUFBEREITEN
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Die Eingabedatei wurde nicht gefunden:\n"
        f"{INPUT_FILE.resolve()}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    TIME_ORDER,
    PERIOD,
    AFD_SUPPORT,
    POSITION_DISTANCE,
    RHETORIC_DISTANCE,
    POSITION_ABSOLUTE,
    RHETORIC_ABSOLUTE,
]

check_required_columns(
    data=df,
    required_columns=required_columns,
)

# Numerische Variablen bereinigen
numeric_columns = [
    PERIOD,
    AFD_SUPPORT,
    POSITION_DISTANCE,
    RHETORIC_DISTANCE,
    POSITION_ABSOLUTE,
    RHETORIC_ABSOLUTE,
]

for column in numeric_columns:
    df[column] = make_numeric(df[column])

# Wahlperiode zunächst numerisch belassen. Die endgültige Umwandlung
# in eine gewöhnliche kategoriale Variable erfolgt nach Listwise
# Deletion innerhalb der Modellfunktion.
df[PERIOD] = pd.to_numeric(
    df[PERIOD],
    errors="coerce",
)

# Datumsvariablen konvertieren
df[TIME_ORDER] = pd.to_datetime(
    df[TIME_ORDER],
    errors="coerce",
)

if NEXT_POLL_DATE in df.columns:
    df[NEXT_POLL_DATE] = pd.to_datetime(
        df[NEXT_POLL_DATE],
        errors="coerce",
    )

if df[TIME_ORDER].isna().all():
    raise ValueError(
        f"Die Zeitvariable '{TIME_ORDER}' konnte nicht als Datum "
        "eingelesen werden."
    )

# Chronologisch sortieren
df = (
    df.sort_values(TIME_ORDER)
    .reset_index(drop=True)
)

# Eindeutigkeit der aggregierten Zeitreihe prüfen
if df[TIME_ORDER].duplicated().any():
    duplicated_dates = (
        df.loc[df[TIME_ORDER].duplicated(keep=False), TIME_ORDER]
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )
    raise ValueError(
        "Die aggregierte Mainstream-Datei enthält mehrere Beobachtungen "
        "für mindestens ein poll_date. Geprüft werden müssen:\n"
        + "\n".join(f"  - {date}" for date in duplicated_dates)
    )

# ============================================================
# 4. ZEITVERZÖGERTE VARIABLEN ERSTELLEN
# ============================================================

# Gemeinsame Kontrollvariable
create_lag(
    data=df,
    source=AFD_SUPPORT,
    target="afd_support_lag1",
)

# Distanzmodelle
create_lag(
    data=df,
    source=RHETORIC_DISTANCE,
    target="rhetoric_distance_lag1",
)
create_lag(
    data=df,
    source=POSITION_DISTANCE,
    target="position_distance_lag1",
)

# Modelle mit absoluten Werten
create_lag(
    data=df,
    source=RHETORIC_ABSOLUTE,
    target="rhetoric_absolute_lag1",
)
create_lag(
    data=df,
    source=POSITION_ABSOLUTE,
    target="position_absolute_lag1",
)

# Diagnosedatei zur Kontrolle der Lag-Zuordnung
diagnostic_columns = [
    INTERVAL_ID,
    TIME_ORDER,
    NEXT_POLL_DATE,
    PERIOD,
    AFD_SUPPORT,
    "afd_support_lag1",
    RHETORIC_DISTANCE,
    "rhetoric_distance_lag1",
    POSITION_DISTANCE,
    "position_distance_lag1",
    RHETORIC_ABSOLUTE,
    "rhetoric_absolute_lag1",
    POSITION_ABSOLUTE,
    "position_absolute_lag1",
]

diagnostic_columns = [
    column for column in diagnostic_columns
    if column in df.columns
]

df[diagnostic_columns].to_csv(
    OUTPUT_DIR / "rq3_rq4_lag_diagnostics.csv",
    index=False,
    float_format="%.3f",
)


# ============================================================
# 5. MODELLDEFINITIONEN
# ============================================================

model_specs = [
    {
        "model_name": "FF3_Distanz",
        "research_question": "FF3",
        "operationalization": "Distanz zur AfD",
        "outcome": POSITION_DISTANCE,
        "predictor_lag": "rhetoric_distance_lag1",
        "outcome_lag": "position_distance_lag1",
        "effect": (
            "rhetorische Distanz zur AfD t-1 "
            "-> positionelle Distanz zur AfD t"
        ),
        "title": (
            "FF3 – Distanzmodell: Einfluss der vorherigen rhetorischen "
            "Distanz auf die nachfolgende positionelle Distanz"
        ),
        "interpretation_type": "distance",
    },
    {
        "model_name": "FF4_Distanz",
        "research_question": "FF4",
        "operationalization": "Distanz zur AfD",
        "outcome": RHETORIC_DISTANCE,
        "predictor_lag": "position_distance_lag1",
        "outcome_lag": "rhetoric_distance_lag1",
        "effect": (
            "positionelle Distanz zur AfD t-1 "
            "-> rhetorische Distanz zur AfD t"
        ),
        "title": (
            "FF4 – Distanzmodell: Einfluss der vorherigen positionellen "
            "Distanz auf die nachfolgende rhetorische Distanz"
        ),
        "interpretation_type": "distance",
    },
    {
        "model_name": "FF3_Absolut",
        "research_question": "FF3",
        "operationalization": "Absolute Mainstream-Werte",
        "outcome": POSITION_ABSOLUTE,
        "predictor_lag": "rhetoric_absolute_lag1",
        "outcome_lag": "position_absolute_lag1",
        "effect": (
            "absoluter Populismuswert t-1 "
            "-> absoluter GAL–TAN-Wert t"
        ),
        "title": (
            "FF3 – Absolutmodell: Einfluss des vorherigen absoluten "
            "Populismuswerts auf den nachfolgenden absoluten GAL–TAN-Wert"
        ),
        "interpretation_type": "absolute",
    },
    {
        "model_name": "FF4_Absolut",
        "research_question": "FF4",
        "operationalization": "Absolute Mainstream-Werte",
        "outcome": RHETORIC_ABSOLUTE,
        "predictor_lag": "position_absolute_lag1",
        "outcome_lag": "rhetoric_absolute_lag1",
        "effect": (
            "absoluter GAL–TAN-Wert t-1 "
            "-> absoluter Populismuswert t"
        ),
        "title": (
            "FF4 – Absolutmodell: Einfluss des vorherigen absoluten "
            "GAL–TAN-Werts auf den nachfolgenden absoluten Populismuswert"
        ),
        "interpretation_type": "absolute",
    },
]


# ============================================================
# 6. MODELLE SCHÄTZEN
# ============================================================

model_results = {}
coefficient_tables = []
focal_tests = []
model_reports = []

for spec in model_specs:
    result, model_df, formula = fit_time_series_model(
        data=df,
        outcome=spec["outcome"],
        predictor_lag=spec["predictor_lag"],
        outcome_lag=spec["outcome_lag"],
        afd_support_lag="afd_support_lag1",
        model_name=spec["model_name"],
    )

    model_results[spec["model_name"]] = {
        "result": result,
        "data": model_df,
        "formula": formula,
        "spec": spec,
    }

    coefficient_tables.append(
        tidy_coefficient_table(
            result=result,
            focal_term=spec["predictor_lag"],
            model_name=spec["model_name"],
            research_question=spec["research_question"],
            operationalization=spec["operationalization"],
        )
    )

    focal_tests.append(
        extract_focal_test(
            result=result,
            focal_term=spec["predictor_lag"],
            research_question=spec["research_question"],
            operationalization=spec["operationalization"],
            effect=spec["effect"],
            n_obs=result.nobs,
        )
    )

    model_reports.append(
        format_model_report(
            title=spec["title"],
            result=result,
            model_df=model_df,
            formula=formula,
            focal_term=spec["predictor_lag"],
            interpretation_type=spec["interpretation_type"],
        )
    )


# ============================================================
# 7. HOLM-KORREKTUREN
# ============================================================

focal_tests_df = pd.DataFrame(focal_tests)

# Holm-Korrektur getrennt innerhalb jeder Operationalisierung:
# jeweils FF3 und FF4 zusammen.
focal_tests_df["p_Holm_family"] = pd.NA
focal_tests_df["significant_Holm_family_005"] = False

for operationalization, indices in focal_tests_df.groupby(
    "operationalization"
).groups.items():
    p_values = focal_tests_df.loc[indices, "p_raw"].astype(float)

    reject, corrected_p, _, _ = multipletests(
        p_values,
        alpha=0.05,
        method="holm",
    )

    focal_tests_df.loc[indices, "p_Holm_family"] = corrected_p
    focal_tests_df.loc[
        indices,
        "significant_Holm_family_005",
    ] = reject

focal_tests_df["p_Holm_family"] = pd.to_numeric(
    focal_tests_df["p_Holm_family"],
    errors="coerce",
)

# Zusätzliche konservative Holm-Korrektur über alle vier Tests
reject_all, corrected_all, _, _ = multipletests(
    focal_tests_df["p_raw"].astype(float),
    alpha=0.05,
    method="holm",
)

focal_tests_df["p_Holm_all"] = corrected_all
focal_tests_df["significant_Holm_all_005"] = reject_all


# ============================================================
# 8. ERGEBNISSE SPEICHERN
# ============================================================

all_coefficients = pd.concat(
    coefficient_tables,
    ignore_index=True,
)

all_coefficients.to_csv(
    OUTPUT_DIR / "rq3_rq4_all_coefficients.csv",
    index=False,
    float_format="%.3f",
)

focal_tests_df.to_csv(
    OUTPUT_DIR / "rq3_rq4_focal_effects_holm.csv",
    index=False,
    float_format="%.3f",
)

# Zusätzliche getrennte Tabellen
distance_focal = focal_tests_df[
    focal_tests_df["operationalization"] == "Distanz zur AfD"
].copy()

absolute_focal = focal_tests_df[
    focal_tests_df["operationalization"] == "Absolute Mainstream-Werte"
].copy()

distance_focal.to_csv(
    OUTPUT_DIR / "rq3_rq4_distance_focal_effects.csv",
    index=False,
    float_format="%.3f",
)

absolute_focal.to_csv(
    OUTPUT_DIR / "rq3_rq4_absolute_focal_effects.csv",
    index=False,
    float_format="%.3f",
)

# Vollständiger Textbericht
with open(
    OUTPUT_DIR / "rq3_rq4_model_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(
        "\n".join(model_reports)
    )

    file.write("\n")
    file.write("=" * 88 + "\n")
    file.write("HOLM-KORRIGIERTE ZENTRALE TESTS\n")
    file.write("=" * 88 + "\n")
    file.write(
        focal_tests_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    file.write("\n\n")

    file.write(
        "Erläuterung der Korrekturvariablen:\n"
        "- p_Holm_family korrigiert FF3 und FF4 getrennt innerhalb "
        "der Distanz- beziehungsweise Absolutmodelle.\n"
        "- p_Holm_all korrigiert konservativ über alle vier zentralen "
        "Tests gemeinsam.\n"
    )


# ============================================================
# 9. KONSOLENAUSGABE
# ============================================================

print("\n" + "=" * 88)
print("FF3 / FF4: KREUZVERZÖGERTE MODELLE AUF MAINSTREAM-EBENE")
print("=" * 88)
print(f"Eingabedatei: {INPUT_FILE}")
print(f"Zeilen im Ausgangsdatensatz: {len(df)}")
print(f"Zeitvariable: {TIME_ORDER}")
print(f"Wahlperioden-FE: {PERIOD}")
print(f"AfD-Kontrollvariable: {AFD_SUPPORT} (verzögert)")
print(f"HAC-/Newey-West maxlags: {HAC_MAXLAGS}")

print("\nVerwendete Distanzvariablen:")
print(f"- Rhetorik: {RHETORIC_DISTANCE}")
print(f"- Position: {POSITION_DISTANCE}")

print("\nVerwendete absolute Variablen:")
print(f"- Rhetorik: {RHETORIC_ABSOLUTE}")
print(f"- Position: {POSITION_ABSOLUTE}")

print("\nZentrale Ergebnisse:")
print(
    focal_tests_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
)

print("\nGespeicherte Dateien:")
for output_path in sorted(OUTPUT_DIR.glob("*")):
    print(f"- {output_path}")