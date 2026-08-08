import os
import json
import time
import random
import hashlib
import datetime as dt
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI


# =====================================================
# EINSTELLUNGEN
# =====================================================

INPUT_FILE = Path("data/processed/03_cppbt_government.csv")

OUTPUT_DIR = Path("data/processed/final_llm_coding")
OUTPUT_JSONL = OUTPUT_DIR / "final_llm_scores.jsonl"
OUTPUT_CSV = OUTPUT_DIR / "final_llm_scores.csv"
OUTPUT_ANALYSIS_READY_CSV = OUTPUT_DIR / "final_llm_scores_analysis_ready.csv"
ERROR_JSONL = OUTPUT_DIR / "final_llm_errors.jsonl"
METADATA_FILE = OUTPUT_DIR / "final_llm_run_metadata.json"

ID_COLUMN = "rede_id"
TEXT_COLUMN = "rede_text"

MODEL = "gpt-4.1-mini"
TEMPERATURE = 0

SAVE_CSV_EVERY = 250
MAX_RETRIES = 6
BASE_SLEEP_SECONDS = 2

RUN_LABEL = "final_main_run_v1"


# =====================================================
# SYSTEM PROMPT
# =====================================================

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


# =====================================================
# USER PROMPT
# =====================================================

USER_TEMPLATE = """

REDE:

\"\"\"{rede}\"\"\"
"""


# =====================================================
# JSON SCHEMA
# =====================================================

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "bundestag_coding",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "P1_people": {
                    "type": "integer",
                    "enum": [0, 1, 2]
                },
                "P2_anti_elite": {
                    "type": "integer",
                    "enum": [0, 1, 2]
                },
                "P3_outgroup": {
                    "type": "integer",
                    "enum": [0, 1, 2]
                },
                "G1_environment": {
                    "type": ["integer", "string"],
                    "enum": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "NaN"]
                },
                "G2_migration": {
                    "type": ["integer", "string"],
                    "enum": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "NaN"]
                },
                "G3_society": {
                    "type": ["integer", "string"],
                    "enum": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "NaN"]
                },
                "G4_civil_rights": {
                    "type": ["integer", "string"],
                    "enum": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "NaN"]
                },
                "G5_cosmopolitanism": {
                    "type": ["integer", "string"],
                    "enum": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "NaN"]
                }
            },
            "required": [
                "P1_people",
                "P2_anti_elite",
                "P3_outgroup",
                "G1_environment",
                "G2_migration",
                "G3_society",
                "G4_civil_rights",
                "G5_cosmopolitanism"
            ]
        }
    }
}


# =====================================================
# VARIABLENLISTEN
# =====================================================

P_COLS = [
    "P1_people",
    "P2_anti_elite",
    "P3_outgroup"
]

G_COLS = [
    "G1_environment",
    "G2_migration",
    "G3_society",
    "G4_civil_rights",
    "G5_cosmopolitanism"
]

ALL_SCORE_COLS = P_COLS + G_COLS


# =====================================================
# HASH-FUNKTIONEN
# =====================================================

def stable_hash(obj):
    """
    Erstellt einen stabilen Hash für Prompt und Schema.
    Dieser Hash dient nur der Dokumentation und Reproduzierbarkeit.
    """

    if isinstance(obj, str):
        text = obj
    else:
        text = json.dumps(
            obj,
            ensure_ascii=False,
            sort_keys=True
        )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


PROMPT_HASH = stable_hash(SYSTEM_PROMPT)
SCHEMA_HASH = stable_hash(SCHEMA)


# =====================================================
# DATEI-HILFSFUNKTIONEN
# =====================================================

def ensure_output_dir():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def utc_now():
    return dt.datetime.now(
        dt.timezone.utc
    ).isoformat()


def append_jsonl(path, record):
    """
    Schreibt einen Datensatz sofort als eigene JSONL-Zeile.
    Dadurch geht bei einem Absturz maximal die gerade laufende Rede verloren.
    """

    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )

        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    """
    Liest eine JSONL-Datei ein.
    Fehlerhafte oder leere Zeilen werden übersprungen.
    """

    if not path.exists():
        return []

    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError:
                print(
                    f"Warnung: Fehlerhafte JSONL-Zeile in {path}, "
                    f"Zeile {line_number}, wurde übersprungen."
                )

    return records


def load_completed_ids():
    """
    Lädt alle bereits erfolgreich codierten Rede-IDs.
    Fehlerhafte Reden werden nicht übersprungen,
    sondern bei einem Neustart erneut versucht.
    """

    records = read_jsonl(OUTPUT_JSONL)

    completed_ids = set()

    for record in records:
        if ID_COLUMN in record:
            completed_ids.add(
                str(record[ID_COLUMN])
            )

    return completed_ids


def export_csv_from_jsonl():
    """
    Erstellt oder aktualisiert eine CSV-Kopie aus der JSONL-Datei.
    Die JSONL-Datei bleibt das primäre Sicherheitsformat.
    """

    records = read_jsonl(OUTPUT_JSONL)

    if not records:
        return

    df_out = pd.DataFrame(records)

    tmp_file = OUTPUT_CSV.with_suffix(".tmp.csv")

    df_out.to_csv(
        tmp_file,
        index=False
    )

    os.replace(
        tmp_file,
        OUTPUT_CSV
    )


def export_analysis_ready_csv():
    """
    Erstellt zusätzlich eine analysefertige CSV:
    - Populismusvariablen als Integer
    - GAL-TAN-Variablen als Float
    - 'NaN'-Strings als echte fehlende Werte
    """

    if not OUTPUT_CSV.exists():
        return

    df = pd.read_csv(
        OUTPUT_CSV,
        keep_default_na=False
    )

    for col in P_COLS:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    for col in G_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .replace("NaN", pd.NA)
                .astype("Float64")
            )

    tmp_file = OUTPUT_ANALYSIS_READY_CSV.with_suffix(".tmp.csv")

    df.to_csv(
        tmp_file,
        index=False
    )

    os.replace(
        tmp_file,
        OUTPUT_ANALYSIS_READY_CSV
    )


def save_or_check_metadata():
    """
    Speichert Metadaten des Laufs.
    Falls die Codierung fortgesetzt wird, prüft diese Funktion,
    ob Modell, Prompt, Schema und Input-Datei unverändert sind.
    """

    metadata = {
        "run_label": RUN_LABEL,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "prompt_hash": PROMPT_HASH,
        "schema_hash": SCHEMA_HASH,
        "input_file": str(INPUT_FILE),
        "id_column": ID_COLUMN,
        "text_column": TEXT_COLUMN,
        "created_or_checked_at_utc": utc_now()
    }

    if METADATA_FILE.exists():

        old_metadata = json.loads(
            METADATA_FILE.read_text(
                encoding="utf-8"
            )
        )

        keys_to_check = [
            "model",
            "temperature",
            "prompt_hash",
            "schema_hash",
            "input_file",
            "id_column",
            "text_column"
        ]

        for key in keys_to_check:

            old_value = old_metadata.get(key)
            new_value = metadata.get(key)

            if old_value != new_value:
                raise ValueError(
                    f"Metadaten-Konflikt bei '{key}'.\n"
                    f"Alt: {old_value}\n"
                    f"Neu: {new_value}\n\n"
                    f"Du versuchst vermutlich, eine bestehende Codierung "
                    f"mit verändertem Prompt, Schema, Modell oder Input fortzusetzen. "
                    f"Nutze dafür einen neuen Output-Ordner oder ein neues RUN_LABEL."
                )

    else:

        METADATA_FILE.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )


# =====================================================
# OPENAI SETUP
# =====================================================

def setup_client():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY wurde nicht gefunden. Prüfe deine .env-Datei."
        )

    client = OpenAI(
        api_key=api_key,
        timeout=120
    )

    return client


client = setup_client()


# =====================================================
# LLM-FUNKTIONEN
# =====================================================

def usage_to_dict(usage):
    """
    Extrahiert Tokeninformationen aus der API-Antwort.
    """

    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None
        }

    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None)
    }


def code_speech(text):
    """
    Codiert eine einzelne Rede mit dem LLM.
    """

    prompt = USER_TEMPLATE.format(
        rede=text
    )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        response_format=SCHEMA,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    message = response.choices[0].message

    if getattr(message, "refusal", None):
        raise ValueError(
            f"Model refusal: {message.refusal}"
        )

    if not message.content:
        raise ValueError(
            "Leere Modellantwort."
        )

    coding = json.loads(
        message.content
    )

    response_meta = {
        "openai_response_id": getattr(response, "id", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        **usage_to_dict(response.usage)
    }

    return coding, response_meta


def code_with_retry(text):
    """
    Wiederholt die Anfrage bei API-, Timeout- oder Parsingfehlern.
    Nutzt exponentielles Warten mit etwas Zufall.
    """

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            return code_speech(text)

        except Exception as e:

            last_error = e

            if attempt == MAX_RETRIES:
                break

            sleep_seconds = min(
                90,
                BASE_SLEEP_SECONDS * (2 ** (attempt - 1))
            )

            sleep_seconds = sleep_seconds + random.uniform(0, 1.5)

            print(
                f"Retry {attempt}/{MAX_RETRIES} nach Fehler: "
                f"{type(e).__name__}: {e}. "
                f"Warte {sleep_seconds:.1f}s."
            )

            time.sleep(sleep_seconds)

    raise last_error


# =====================================================
# DATENPRÜFUNGEN
# =====================================================

def validate_input_df(df):
    """
    Prüft, ob die notwendigen Spalten vorhanden sind
    und ob die Rede-IDs eindeutig sind.
    """

    required_cols = [
        ID_COLUMN,
        TEXT_COLUMN
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Fehlende Spalten im Input: {missing_cols}"
        )

    duplicated = df[ID_COLUMN].astype(str).duplicated().sum()

    if duplicated > 0:
        raise ValueError(
            f"{duplicated} doppelte rede_id-Werte gefunden. "
            f"Für die Resume-Logik müssen Rede-IDs eindeutig sein."
        )


def is_valid_text(text):
    """
    Prüft, ob ein Redetext codierbar ist.
    """

    if text is None:
        return False

    text = str(text).strip()

    if text == "":
        return False

    if text.lower() in ["nan", "none", "null"]:
        return False

    return True


# =====================================================
# MAIN
# =====================================================

def main():
    ensure_output_dir()
    save_or_check_metadata()

    df = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False
    )

    validate_input_df(df)

    df[ID_COLUMN] = df[ID_COLUMN].astype(str)

    completed_ids = load_completed_ids()

    todo_df = df[
        ~df[ID_COLUMN].isin(completed_ids)
    ].copy()

    print("======================================")
    print("FINALE LLM-CODIERUNG")
    print("======================================")
    print(f"Input-Datei: {INPUT_FILE}")
    print(f"Output-Ordner: {OUTPUT_DIR}")
    print(f"Modell: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Prompt-Hash: {PROMPT_HASH}")
    print(f"Schema-Hash: {SCHEMA_HASH}")
    print("--------------------------------------")
    print(f"Input-Reden insgesamt: {len(df)}")
    print(f"Bereits erfolgreich codiert: {len(completed_ids)}")
    print(f"Noch zu codieren: {len(todo_df)}")
    print("======================================")

    successful_since_csv_export = 0
    successful_this_run = 0
    errors_this_run = 0

    for _, row in tqdm(
        todo_df.iterrows(),
        total=len(todo_df)
    ):

        speech_id = str(row[ID_COLUMN])
        text = row[TEXT_COLUMN]

        if not is_valid_text(text):

            error_record = {
                ID_COLUMN: speech_id,
                "run_label": RUN_LABEL,
                "model": MODEL,
                "prompt_hash": PROMPT_HASH,
                "schema_hash": SCHEMA_HASH,
                "failed_at_utc": utc_now(),
                "error_type": "InvalidText",
                "error": "Leerer oder ungültiger Redetext."
            }

            append_jsonl(
                ERROR_JSONL,
                error_record
            )

            errors_this_run += 1

            continue

        try:

            coding, response_meta = code_with_retry(
                str(text)
            )

            result_record = {
                ID_COLUMN: speech_id,
                "run_label": RUN_LABEL,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "prompt_hash": PROMPT_HASH,
                "schema_hash": SCHEMA_HASH,
                "coded_at_utc": utc_now(),
                **coding,
                **response_meta
            }

            append_jsonl(
                OUTPUT_JSONL,
                result_record
            )

            successful_this_run += 1
            successful_since_csv_export += 1

            if successful_since_csv_export >= SAVE_CSV_EVERY:

                export_csv_from_jsonl()
                export_analysis_ready_csv()

                successful_since_csv_export = 0

        except Exception as e:

            error_record = {
                ID_COLUMN: speech_id,
                "run_label": RUN_LABEL,
                "model": MODEL,
                "prompt_hash": PROMPT_HASH,
                "schema_hash": SCHEMA_HASH,
                "failed_at_utc": utc_now(),
                "error_type": type(e).__name__,
                "error": str(e)
            }

            append_jsonl(
                ERROR_JSONL,
                error_record
            )

            errors_this_run += 1

    export_csv_from_jsonl()
    export_analysis_ready_csv()

    print("\n======================================")
    print("FERTIG")
    print("======================================")
    print(f"Erfolgreich in diesem Lauf: {successful_this_run}")
    print(f"Fehler in diesem Lauf: {errors_this_run}")
    print(f"Primäre Ergebnisdatei JSONL: {OUTPUT_JSONL}")
    print(f"CSV-Kopie: {OUTPUT_CSV}")
    print(f"Analysefertige CSV: {OUTPUT_ANALYSIS_READY_CSV}")
    print(f"Fehlerlog: {ERROR_JSONL}")
    print("======================================")


if __name__ == "__main__":
    main()