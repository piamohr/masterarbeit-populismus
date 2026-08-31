import json
import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# =====================================================
# EINSTELLUNGEN
# =====================================================

INPUT_FILE = "data/processed/validation/validation_sample.csv"

OUTPUT_FILE = "data/processed/validation/validation_llm_scores.csv"
ERROR_FILE = "data/processed/validation/validation_llm_errors.csv"

ID_COLUMN = "rede_id"
TEXT_COLUMN = "rede_text"

MODEL = "gpt-4.1-mini"

SAVE_EVERY = 50
MAX_RETRIES = 3

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

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
-	Politische Eliten (Regierung, Parteien, „das Establishment")
-	Wirtschaftliche Eliten (Konzerne, Banken, „die da oben")
-	Medieneliten („Mainstream-Medien", „Lügenpresse")
-	Intellektuelle/kulturelle Eliten
-	Supranationale Eliten (EU, internationale Organisationen)

0 = fehlt
1 = teilweise
2 = dominant

P3_outgroup:
Die Rede stigmatisiert bestimmte Bevölkerungsgruppen als „gefährliche Andere", die nicht zum „Volk" gehören und eine Bedrohung darstellen.

-	Gefährliche Andere: Bestimmte Gruppen (z.B. Migranten, Geflüchtete, Bürgergeldempfänger, religiöse oder ethnische Minderheiten, andere Länder) werden als Bedrohung für das Volk dargestellt und als Sündenböcke instrumentalisiert.
-	Autoritarismus: Forderungen nach illiberalen oder harten Maßnahmen gegen Gruppen, die die Homogenität des Volkes bedrohen.

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
# LLM FUNKTION
# =====================================================

def code_speech(text):

    prompt = USER_TEMPLATE.format(
        rede=text
    )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
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

    return json.loads(
        response.choices[0].message.content
    )


def code_with_retry(text):

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:
            return code_speech(text)

        except Exception as e:

            last_error = e

            print(
                f"Retry {attempt+1}: {e}"
            )

            time.sleep(2)

    raise last_error

# =====================================================
# MAIN
# =====================================================

def main():

    df = pd.read_csv(INPUT_FILE)

    print(f"{len(df)} Reden geladen")
    print(df[[ID_COLUMN]].head())

    results = []
    errors = []

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df)
    ):

        speech_id = row[ID_COLUMN]
        text = row[TEXT_COLUMN]

        try:

            coding = code_with_retry(
                str(text)
            )

            results.append({
                ID_COLUMN: speech_id,
                **coding
            })

        except Exception as e:

            errors.append({
                ID_COLUMN: speech_id,
                "error": str(e)
            })

        if len(results) % SAVE_EVERY == 0:

            pd.DataFrame(
                results
            ).to_csv(
                OUTPUT_FILE,
                index=False
            )

    pd.DataFrame(results).to_csv(
        OUTPUT_FILE,
        index=False
    )

    pd.DataFrame(errors).to_csv(
        ERROR_FILE,
        index=False
    )

    print("Fertig.")


if __name__ == "__main__":
    main()