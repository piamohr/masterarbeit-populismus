# Masterarbeit: Die zwei Gesichter des populistischen Zeitgeistes: Welchen Einfluss hat der AfD-Erfolg auf den deutschen Parteienwettbewerb?

Dieses Repository enthält die Python-Skripte und zentrale Datendateien für eine quantitative Analyse von Bundestagsreden. Untersucht wird, ob und wie sich Mainstream-Parteien rhetorisch und positionell an die AfD annähern. Die Analyse kombiniert LLM-basierte Inhaltscodierung, Umfragedaten, Aggregationen auf Umfrageintervall-Ebene sowie Regressionsmodelle zu den Hypothesen und Forschungsfragen der Masterarbeit.

## Forschungsinteresse

Ziel des Projekts ist es, den Einfluss des Erfolgs der AfD auf die Kommunikation und Themenpositionierung von Mainstream-Parteien zu untersuchen. Im Mittelpunkt stehen zwei Formen der Annäherung:

1. **Positionelle Annäherung**: Veränderungen der Distanz von Mainstream-Parteien zur AfD auf der GAL-TAN-Dimension.
2. **Rhetorische Annäherung**: Veränderungen populistischer Kommunikation von Mainstream-Parteien.

Die zentralen Hypothesen prüfen, ob höhere AfD-Unterstützung mit geringeren positionellen Distanzen und höherer populistischer Kommunikation der Mainstream-Parteien zusammenhängt. Ergänzend werden Moderationseffekte durch ideologische Nähe, Stimmenverluste und Regierungs-/Oppositionsstatus sowie Unterschiede zwischen Subdimensionen untersucht.

## Datenbasis

Die Bundestagsreden stammen aus dem **Corpus der Plenarprotokolle des Deutschen Bundestages (CPP-BT)** von Sean Fobbe. Verwendet wird die speech-level Parquet-Datei:

```text
CPP-BT_2026-01-17_DE_PQT_Reden_Gesamt.parquet
```

Diese Datei ist wegen ihrer Größe **nicht im Repository enthalten**. Für eine vollständige Reproduktion der Datenaufbereitung muss sie separat heruntergeladen und lokal unter folgendem Pfad abgelegt werden:

```text
data/raw/CPP-BT_2026-01-17_DE_PQT_Reden_Gesamt.parquet
```

Die verwendete Reden-Variante enthält Einzelreden und Metadaten wie `rede_id`, `rede_text`, `wahlperiode`, `sitzung_datum`, `redner_fraktion`, `redner_rolle_kurz` und `redner_rolle_lang`.

Zusätzlich werden Umfragedaten von Infratest dimap sowie manuelle Validierungsdaten verwendet. Diese kleineren Rohdatendateien werden im Repository mitgeführt:

```text
data/raw/infratest_dimap.xlsx
data/raw/manual_validation.csv
data/raw/manual_validation.xlsx
```

## Projektstruktur

```text
.
├── src/                              # Python-Skripte der Analysepipeline
├── data/
│   ├── raw/                          # kleinere Rohdaten auf GitHub; große CPP-BT-Parquet-Datei lokal
│   ├── interim/                      # optionale Zwischendaten, nicht auf GitHub
│   └── processed/
│       └── final_analysis/           # finale Analysedatensätze, auf GitHub
├── results/                          # Modelloutputs, Tabellen und Abbildungen, auf GitHub
├── outputs/                          # Hilfsoutputs, nicht auf GitHub
├── requirements.txt                  # Laufzeit-Abhängigkeiten
├── requirements-dev.txt              # optionale Entwicklungs-Abhängigkeiten
├── .gitignore
└── README.md
```

## Installation

Empfohlen wird eine virtuelle Umgebung.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Unter Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Für die Arbeit mit Jupyter oder Formatierungswerkzeugen können zusätzlich die Entwicklungsabhängigkeiten installiert werden:

```bash
pip install -r requirements-dev.txt
```

## API-Key für LLM-Codierung

Für die LLM-Codierung wird ein OpenAI-API-Key benötigt. Dieser wird **nicht** ins Repository aufgenommen, sondern lokal in einer `.env`-Datei gespeichert:

```text
OPENAI_API_KEY=dein_api_key
```

Die Datei `.env` ist in `.gitignore` ausgeschlossen. Eine `.env.example` ist im Repository hinterlegt.

Wichtig: Die LLM-Codierungsskripte sollten nur erneut ausgeführt werden, wenn eine erneute Codierung wirklich beabsichtigt ist. Für Replikationen der nachgelagerten Analysen können die bereits erzeugten LLM-Scores bzw. finalen Analysedatensätze verwendet werden.

## Pipeline und Skriptreihenfolge

### 1. Datenaufbereitung CPP-BT

```text
01_load_cppbt.py
02_clean_cppbt.py
05_add_government_status_cppbt.py
06_test_sample.py
07_print_speeches.py
```

`03_describe_cppbt.py` und `04_check_random_speeches.py` wurden aus der bereinigten Pipeline entfernt, weil sie für die finale Reproduktion nicht benötigt werden.

### 2. Validierung der LLM-Codierung

```text
11_validation_sample.py
12_validation_llm_coding.py
13_face_validation.py
14_concurrent_validation.py
```

`12_validation_llm_coding.py` enthält die LLM-basierte Codierung des Validierungssamples. Prompt, JSON-Schema, Modellparameter und Parsing-Logik sollten nicht verändert werden, wenn Vergleichbarkeit mit den bestehenden Ergebnissen erhalten bleiben soll.

### 3. Finale LLM-Codierung und Score-Bildung

```text
15_final_llm_coding.py
16_scores.py
17_merging.py
```

`15_final_llm_coding.py` codiert die finale Redestichprobe. `16_scores.py` berechnet Populismus- und GAL-TAN-Scores. `17_merging.py` verknüpft die ursprünglichen Rededaten mit den LLM-Scores.

### 4. Umfrageintervalle und deskriptive Statistiken

```text
18_polling_intervals.py
19_deskriptive_statistiken.py
```

Diese Skripte ordnen Reden Umfrageintervallen zu, aggregieren Daten auf Partei- und Mainstream-Ebene und erzeugen deskriptive Statistiken.
Hinweis: Die Datei `data/processed/final_analysis/02_speech_level_with_polling.csv` ist aufgrund der GitHub-Dateigrößenbeschränkung nicht im Repository enthalten. Sie ist eine generierte Zwischen-/Analysedatei und kann durch Ausführen der Preprocessing- und Merging-Pipeline reproduziert werden. Die kleineren aggregierten finalen Analysedatensätze in `data/processed/final_analysis/` sind im Repository enthalten.

### 5. H1 bis H1c: positionelle Annäherung

```text
20_hypothesis1.py
21_hypothesis1_mainstream.py
22_hypothesis1a.py
23_hypothesis1b.py
24_hypothesis1c.py
```

Diese Skripte prüfen den Zusammenhang zwischen AfD-Unterstützung und positioneller Annäherung auf der GAL-TAN-Dimension. Enthalten sind Modelle auf Partei-Intervall-Ebene sowie aggregierter Mainstream-Ebene und Moderationsanalysen.

### 6. FF1

```text
25_research_question1.py
```

FF1 untersucht, ob sich GAL-TAN-Subdimensionen hinsichtlich ihres Zusammenhangs mit der AfD-Unterstützung unterscheiden.

### 7. H2 bis H2c: rhetorische Annäherung

```text
27_hypothesis2.py
28_hypothesis2_mainstream.py
29_hypothesis2a.py
30_hypothesis2b.py
31_hypothesis2c.py
```

Diese Skripte prüfen den Zusammenhang zwischen AfD-Unterstützung und populistischer Kommunikation der Mainstream-Parteien.

### 8. FF2, FF3/FF4, AfD-Zusatzanalyse und finale Abbildungen

```text
32_research_question2.py
33_research_question3_4.py
34_afd_check.py
35_figures.py
```

FF2 untersucht Unterschiede zwischen Merkmalen populistischer Kommunikation. FF3/FF4 analysieren das Verhältnis rhetorischer und inhaltlicher Annäherung. `34_afd_check.py` prüft ergänzend, ob sich auch die AfD selbst über die Zeit bzw. mit zunehmender Unterstützung verändert. `35_figures.py` erzeugt finale Abbildungen.

## Zentrale finale Analysedatensätze

Die wichtigsten finalen Datensätze liegen in:

```text
data/processed/final_analysis/
```

Besonders relevant sind:

```text
02_speech_level_with_polling.csv
05_party_polling_interval_distances.csv
06_mainstream_polling_interval_distances.csv
```

Diese Dateien werden bewusst auf GitHub versioniert, da sie die Grundlage der finalen Analysen bilden.

## Ergebnisse

Die Ergebnisdateien liegen in:

```text
results/
```

Auch dieser Ordner wird bewusst auf GitHub versioniert. Er enthält Modelloutputs, Tabellen und Abbildungen zu den Hypothesen und Forschungsfragen.

## Hinweise zur Reproduktion

Für eine vollständige Reproduktion der gesamten Pipeline muss die große CPP-BT-Parquet-Datei lokal ergänzt werden:

```text
data/raw/CPP-BT_2026-01-17_DE_PQT_Reden_Gesamt.parquet
```

Die kleineren Rohdatendateien `infratest_dimap.xlsx`, `manual_validation.csv` und `manual_validation.xlsx` werden im Repository mitgeführt. Die LLM-Skripte können API-Kosten verursachen und durch Modellversionen oder API-Verhalten beeinflusst werden. Für die Reproduktion der statistischen Analysen empfiehlt es sich daher, die versionierten finalen Analysedatensätze und Ergebnisdateien zu verwenden, sofern keine erneute Codierung geplant ist.

## GitHub-Hinweise

Dieses Repository ist so angelegt, dass sensible Dateien, lokale Umgebungen, Hilfsoutputs und große Rohdaten ausgeschlossen bleiben, während kleinere Rohdaten, finale Analysedatensätze und Ergebnisse versioniert werden:

- `.env`, `.venv/`, `data/interim/`, sonstige `data/processed/`-Dateien, `outputs/` und große Parquet-Dateien in `data/raw/` werden ignoriert.
- `data/raw/infratest_dimap.xlsx`, `data/raw/manual_validation.csv`, `data/raw/manual_validation.xlsx`, `data/processed/final_analysis/` und `results/` können auf GitHub hochgeladen werden.