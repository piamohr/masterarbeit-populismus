from pathlib import Path
import json
import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


INPUT_FILE = Path("data/processed/validation/validation_sample.csv")
OUTPUT_FILE_LONG = Path("data/processed/validation/validation_llm_scores_10runs_long.csv")
OUTPUT_FILE_STABILITY = Path("data/processed/validation/validation_llm_stability_report.csv")
OUTPUT_FILE_AGGREGATED = Path("data/processed/validation/validation_llm_scores_10runs_aggregated.csv")
ERROR_FILE = Path("data/processed/validation/validation_llm_errors_10runs.csv")

ID_COLUMN = "rede_id"
TEXT_COLUMN = "rede_text"

MODEL = "gpt-4.1-mini"
TEMPERATURE = 0
N_RUNS = 10
SAVE_EVERY = 50
MAX_RETRIES = 3

# Kein Seed setzen: Ziel ist echte Run-zu-Run-Stabilität.

P_COLS = [
    "P1_people",
    "P2_anti_elite",
    "P3_outgroup",
]

G_COLS = [
    "G1_environment",
    "G2_migration",
    "G3_society",
    "G4_civil_rights",
    "G5_cosmopolitanism",
]

ALL_SCORE_COLS = P_COLS + G_COLS
REQUIRED_COLUMNS = [ID_COLUMN, TEXT_COLUMN]


SYSTEM_PROMPT = """
Du bist wissenschaftlicher Codierer für politische Kommunikationsforschung.

Codiere Bundestagsreden ausschließlich anhand des Textes.
Nutze keine Annahmen über Partei, Sprecher oder Zeitraum.

Du analysierst eine Bundestagsrede.

Aufgabe: Vergib Scores für populistische Rhetorik (P1–P3) und GAL-TAN-Positionen (G1–G5). Bei Unsicherheit zwischen zwei numerischen Intensitäten: wähle den niedrigeren Wert. Lasse kein Feld leer.

POPULISTISCHE RHETORIK, Score 0–2:

Populismus ist eine dünne Ideologie, die die Gesellschaft in zwei  antagonistische Gruppen teilt – das tugendhafte Volk und die korrupte Elite – und  argumentiert, dass Politik Ausdruck des allgemeinen Volkswillens sein sollte.
WICHTIG: Populistische Rhetorik unterscheidet sich von sachlicher Kritik durch einen MORALISIERENDEN Rahmen. Es geht nicht um rationale Kritik an konkreten Missständen, sondern um eine grundsätzliche moralische Verurteilung: Das Volk ist gut und rein, die Elite ist böse und verdorben. Nur wenn dieser moralisierende Rahmen vorhanden ist, handelt es sich um populistische Rhetorik.

P1_people:
Sprechen im Namen des Volkes und Konstruktion eines homogenen, tugendhaften, fleißigen, moralisch legitimierten Volkes mit einem gemeinsamen Willen oder gemeinsamen Interessen. Allein die Missstände anzusprechen, ist NICHT populistisch.

0 = fehlt
1 = teilweise
2 = dominant

P2_anti_elite:
Eliten (z.B. Regierung, Parteien, EU, Medien, Gerichte, wirtschaftliche Eliten, Intellektuelle, kapitalistische Akteure) werden als gegen den Volkswillen handelnd und als korrupt oder eigennützig dargestellt. Sachliche Kritik ohne Moralisierung ist nicht populistisch.

Beispiele:
- Politische Eliten (Regierung, Parteien, „das Establishment")
- Wirtschaftliche Eliten (Konzerne, Banken, „die da oben")
- Medieneliten („Mainstream-Medien", „Lügenpresse")
- Intellektuelle/kulturelle Eliten
- Supranationale Eliten (EU, internationale Organisationen)

0 = fehlt
1 = teilweise
2 = dominant

P3_outgroup:
Die Rede stigmatisiert bestimmte Bevölkerungsgruppen als „gefährliche Andere", die nicht zum „Volk" gehören und eine Bedrohung darstellen.

- Gefährliche Andere: Bestimmte Gruppen (z.B. Migranten, Geflüchtete, Bürgergeldempfänger, religiöse oder ethnische Minderheiten, andere Länder) werden als Bedrohung für das Volk dargestellt und als Sündenböcke instrumentalisiert.
- Autoritarismus: Forderungen nach illiberalen oder harten Maßnahmen gegen Gruppen, die die Homogenität des Volkes bedrohen.

0 = fehlt
1 = teilweise
2 = dominant

GAL-TAN-POSITIONEN

Codiere nur explizite oder klar implizite Positionen. Wenn eine GAL-TAN-Dimension überhaupt nicht thematisiert wird, gib "NaN" aus. Lasse kein Feld leer.

0 = stark GAL, 10 = stark TAN

G1_environment:
0 = Klimaschutz und Nachhaltigkeit haben Vorrang; ökologische Transformation der Wirtschaft; 10 = Wirtschaftswachstum hat Vorrang vor Umweltschutz; Skepsis gegenüber Klimapolitik.

G2_migration:
0 = offene, humanitäre, multikulturelle Position, humanitäre Aufnahmepflicht; 10 = restriktive Migration, Leitkultur, Begrenzung, nationale Zugehörigkeit.

G3_society:
0 = progressive gesellschaftliche Werte, Förderung von LGBTQ+-Rechten, Gleichstellung, Diversität, Minderheitenrechte; 10 = traditionelle Werte/Rollenbilder, Ablehnung progressiver Gesellschaftspolitik.

G4_civil_rights:
0 = Liberale Freiheitsrechte, Grundrechte, Bürgerfreiheiten, rechtsstaatliche Garantien; 10 = Law-and-Order, Sicherheit und staatliche Autorität auch auf Kosten individueller Freiheit.

G5_cosmopolitanism:
0 = internationale Kooperation, Multilateralismus, globale Solidarität, EU-Solidarität; 10 = nationale Interessen, Nationalismus, Skepsis gegenüber internationalen Verpflichtungen, EU-Skepsis.
"""

USER_TEMPLATE = """
REDE:

\"\"\"{rede}\"\"\"
"""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "bundestag_coding",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "P1_people": {"type": "integer", "enum": [0, 1, 2]},
                "P2_anti_elite": {"type": "integer", "enum": [0, 1, 2]},
                "P3_outgroup": {"type": "integer", "enum": [0, 1, 2]},
                "G1_environment": {"type": ["integer", "string"], "enum": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "NaN"]},
                "G2_migration": {"type": ["integer", "string"], "enum": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "NaN"]},
                "G3_society": {"type": ["integer", "string"], "enum": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "NaN"]},
                "G4_civil_rights": {"type": ["integer", "string"], "enum": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "NaN"]},
                "G5_cosmopolitanism": {"type": ["integer", "string"], "enum": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "NaN"]},
            },
            "required": ALL_SCORE_COLS,
        },
    },
}


def check_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "Folgende erwartete Spalten fehlen im Input-Datensatz: "
            + ", ".join(missing_columns)
        )


def create_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY wurde nicht gefunden. Prüfe deine .env-Datei.")

    return OpenAI(api_key=api_key)


def code_speech(client: OpenAI, text: str) -> dict:
    prompt = USER_TEMPLATE.format(rede=text)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        response_format=SCHEMA,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return json.loads(response.choices[0].message.content)


def code_with_retry(client: OpenAI, text: str) -> dict:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return code_speech(client, text)
        except Exception as error:
            last_error = error
            print(f"Retry {attempt}/{MAX_RETRIES}: {error}")
            time.sleep(2)

    raise last_error


def is_present(value: object) -> bool:
    """True, wenn eine GAL-TAN-Dimension numerisch codiert wurde."""
    return str(value) != "NaN"


def calculate_pairwise_percent_agreement(results_df: pd.DataFrame, col: str) -> tuple[float | None, int]:
    """
    Berechnet für jede Rede die paarweise Übereinstimmung über alle Runs und
    mittelt anschließend über alle Reden.
    """
    agreements = []

    for _, group in results_df.groupby(ID_COLUMN):
        values = group.sort_values("run")[col].tolist()

        if len(values) < 2:
            continue

        n_pairs = 0
        n_agree = 0

        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                n_pairs += 1
                n_agree += int(values[i] == values[j])

        agreements.append(n_agree / n_pairs)

    if not agreements:
        return None, 0

    return sum(agreements) / len(agreements), len(agreements)


def calculate_populism_stability(results_df: pd.DataFrame) -> list[dict]:
    rows = []

    for col in P_COLS:
        complete_agreements = []
        modal_agreements = []

        for _, group in results_df.groupby(ID_COLUMN):
            values = group[col].dropna().tolist()

            if len(values) < 2:
                continue

            complete_agreements.append(int(len(set(values)) == 1))

            modal_share = pd.Series(values).value_counts(normalize=True).iloc[0]
            modal_agreements.append(modal_share)

        pairwise_agreement, n_pairwise = calculate_pairwise_percent_agreement(results_df, col)

        rows.extend(
            [
                {
                    "variable": col,
                    "type": "populism",
                    "metric": "share_all_runs_identical",
                    "value": round(sum(complete_agreements) / len(complete_agreements), 3) if complete_agreements else None,
                    "n_speeches": len(complete_agreements),
                },
                {
                    "variable": col,
                    "type": "populism",
                    "metric": "mean_modal_agreement",
                    "value": round(sum(modal_agreements) / len(modal_agreements), 3) if modal_agreements else None,
                    "n_speeches": len(modal_agreements),
                },
                {
                    "variable": col,
                    "type": "populism",
                    "metric": "mean_pairwise_percent_agreement",
                    "value": round(pairwise_agreement, 3) if pairwise_agreement is not None else None,
                    "n_speeches": n_pairwise,
                },
            ]
        )

    return rows


def calculate_gal_tan_stability(results_df: pd.DataFrame) -> list[dict]:
    rows = []

    for col in G_COLS:
        presence_all_identical = []
        presence_modal_agreements = []
        numeric_sds = []
        numeric_ranges = []

        for _, group in results_df.groupby(ID_COLUMN):
            values = group[col].tolist()

            if len(values) < 2:
                continue

            present_values = [is_present(value) for value in values]
            presence_all_identical.append(int(len(set(present_values)) == 1))

            presence_modal_share = pd.Series(present_values).value_counts(normalize=True).iloc[0]
            presence_modal_agreements.append(presence_modal_share)

            numeric_values = [float(value) for value in values if is_present(value)]

            if len(numeric_values) >= 2:
                numeric_sds.append(pd.Series(numeric_values).std())
                numeric_ranges.append(max(numeric_values) - min(numeric_values))

        pairwise_agreement, n_pairwise = calculate_pairwise_percent_agreement(results_df, col)

        rows.extend(
            [
                {
                    "variable": col,
                    "type": "gal-tan",
                    "metric": "share_presence_all_runs_identical",
                    "value": round(sum(presence_all_identical) / len(presence_all_identical), 3) if presence_all_identical else None,
                    "n_speeches": len(presence_all_identical),
                },
                {
                    "variable": col,
                    "type": "gal-tan",
                    "metric": "mean_presence_modal_agreement",
                    "value": round(sum(presence_modal_agreements) / len(presence_modal_agreements), 3) if presence_modal_agreements else None,
                    "n_speeches": len(presence_modal_agreements),
                },
                {
                    "variable": col,
                    "type": "gal-tan",
                    "metric": "mean_numeric_sd",
                    "value": round(sum(numeric_sds) / len(numeric_sds), 3) if numeric_sds else None,
                    "n_speeches": len(numeric_sds),
                },
                {
                    "variable": col,
                    "type": "gal-tan",
                    "metric": "mean_numeric_range",
                    "value": round(sum(numeric_ranges) / len(numeric_ranges), 3) if numeric_ranges else None,
                    "n_speeches": len(numeric_ranges),
                },
                {
                    "variable": col,
                    "type": "gal-tan",
                    "metric": "mean_pairwise_exact_agreement",
                    "value": round(pairwise_agreement, 3) if pairwise_agreement is not None else None,
                    "n_speeches": n_pairwise,
                },
            ]
        )

    return rows


def calculate_stability(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.extend(calculate_populism_stability(results_df))
    rows.extend(calculate_gal_tan_stability(results_df))
    return pd.DataFrame(rows)


def aggregate_10_runs(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert erfolgreiche Runs je Rede.

    P1-P3:
        Modalwert über alle erfolgreichen Runs.

    G1-G5:
        Bei Presence in mindestens 50 % der erfolgreichen Runs wird der Mittelwert
        der numerischen Werte gebildet. Sonst wird 'NaN' ausgegeben.
    """
    aggregated_rows = []

    for speech_id, group in results_df.groupby(ID_COLUMN):
        row = {
            ID_COLUMN: speech_id,
            "n_runs_successful": len(group),
        }

        for col in P_COLS:
            values = group[col].dropna().tolist()

            if values:
                value_counts = pd.Series(values).value_counts(normalize=True)
                mode_value = pd.Series(values).mode().iloc[0]

                row[col] = int(mode_value)
                row[f"{col}_modal_share"] = round(value_counts.iloc[0], 3)
                row[f"{col}_all_runs_identical"] = int(len(set(values)) == 1)
            else:
                row[col] = pd.NA
                row[f"{col}_modal_share"] = pd.NA
                row[f"{col}_all_runs_identical"] = pd.NA

        for col in G_COLS:
            values = group[col].tolist()
            present_values = [is_present(value) for value in values]
            presence_share = sum(present_values) / len(present_values) if present_values else 0
            numeric_values = [float(value) for value in values if is_present(value)]

            row[f"{col}_presence_share"] = round(presence_share, 3)

            if presence_share >= 0.5 and numeric_values:
                row[col] = round(sum(numeric_values) / len(numeric_values), 3)
                row[f"{col}_numeric_sd"] = round(pd.Series(numeric_values).std(), 3) if len(numeric_values) >= 2 else 0
                row[f"{col}_numeric_range"] = round(max(numeric_values) - min(numeric_values), 3)
            else:
                row[col] = "NaN"
                row[f"{col}_numeric_sd"] = pd.NA
                row[f"{col}_numeric_range"] = pd.NA

        aggregated_rows.append(row)

    return pd.DataFrame(aggregated_rows)


def save_long_results(results: list[dict], errors: list[dict]) -> None:
    OUTPUT_FILE_LONG.parent.mkdir(parents=True, exist_ok=True)
    ERROR_FILE.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(OUTPUT_FILE_LONG, index=False)
    pd.DataFrame(errors).to_csv(ERROR_FILE, index=False)


def main() -> None:
    client = create_client()

    df = pd.read_csv(INPUT_FILE)
    check_required_columns(df, REQUIRED_COLUMNS)

    print(f"{len(df)} Reden geladen")
    print(df[[ID_COLUMN]].head())

    results = []
    errors = []
    total_calls = len(df) * N_RUNS

    for run in range(1, N_RUNS + 1):
        print("\n======================================")
        print(f"Starte Durchlauf {run}/{N_RUNS}")
        print("======================================\n")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            speech_id = row[ID_COLUMN]
            text = str(row[TEXT_COLUMN])

            try:
                coding = code_with_retry(client, text)
                results.append({ID_COLUMN: speech_id, "run": run, **coding})
            except Exception as error:
                errors.append({ID_COLUMN: speech_id, "run": run, "error": str(error)})

            if len(results) > 0 and len(results) % SAVE_EVERY == 0:
                save_long_results(results, errors)
                print(
                    f"Zwischengespeichert: {len(results)} erfolgreiche Codierungen "
                    f"von ca. {total_calls} geplanten API-Calls."
                )

    results_df = pd.DataFrame(results)
    errors_df = pd.DataFrame(errors)

    save_long_results(results, errors)

    print("\nCodierung fertig.")
    print(f"Erfolgreiche Codierungen: {len(results_df)}")
    print(f"Fehler: {len(errors_df)}")

    print("\nBerechne Stabilitätsreport...")
    stability_df = calculate_stability(results_df)

    OUTPUT_FILE_STABILITY.parent.mkdir(parents=True, exist_ok=True)
    stability_df.to_csv(OUTPUT_FILE_STABILITY, index=False)

    print(f"\nStabilitätsreport gespeichert: {OUTPUT_FILE_STABILITY}")
    print(stability_df)

    print("\nErstelle aggregierte 10-Run-Datei...")
    aggregated_df = aggregate_10_runs(results_df)

    OUTPUT_FILE_AGGREGATED.parent.mkdir(parents=True, exist_ok=True)
    aggregated_df.to_csv(OUTPUT_FILE_AGGREGATED, index=False)

    print(f"\nAggregierte Datei gespeichert: {OUTPUT_FILE_AGGREGATED}")
    print("\nFertig.")


if __name__ == "__main__":
    main()
