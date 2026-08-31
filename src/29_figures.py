# ============================================================
# APA-7-ABBILDUNGEN: H1a/H2a, FF1/FF2 UND ZEITVERLÄUFE
#
# Erzeugte Abbildungen:
# 1. figure_h1a_interaction.png
#    Links: Distanzwerte | Rechts: absolute Werte
# 2. figure_h2a_interaction.png
#    Links: Distanzwerte | Rechts: absolute Werte
# 3. figure_ff1_pairwise_matrix.png
#    Links: Distanzwerte | Rechts: absolute Werte
# 4. figure_ff2_pairwise_matrix.png
#    Links: Distanzwerte | Rechts: absolute Werte
# 5. figure_gal_tan_scores_and_afd_poll_over_time.png
# 6. figure_populism_scores_and_afd_poll_over_time.png
#
# Ausgabe ausschließlich als PNG mit 300 dpi.
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. DATEIPFADE
# ============================================================

RESULTS_DIR = Path("results")
FIGURE_DIR = RESULTS_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

PARTY_INTERVAL_FILE = Path(
    "data/processed/final_analysis/03_party_polling_interval.csv"
)

# ------------------------------------------------------------
# H1a/H2a: Vorhersagedateien
# ------------------------------------------------------------

H1A_DISTANCE_PREDICTION_FILE = (
    RESULTS_DIR / "hypothesis1a" / "h1a_distance_predicted_values.csv"
)
H1A_ABSOLUTE_PREDICTION_FILE = (
    RESULTS_DIR / "hypothesis1a" / "h1a_absolute_gal_tan_predicted_values.csv"
)
H2A_DISTANCE_PREDICTION_FILE = (
    RESULTS_DIR / "hypothesis2a" / "h2a_distance_predicted_values.csv"
)
H2A_ABSOLUTE_PREDICTION_FILE = (
    RESULTS_DIR / "hypothesis2a" / "h2a_absolute_populism_predicted_values.csv"
)

# ------------------------------------------------------------
# FF1/FF2: Paarweise Vergleiche
# ------------------------------------------------------------

FF1_DISTANCE_PAIRWISE_FILE = (
    RESULTS_DIR / "research_question1" / "rq1_distance_pairwise_dimension_tests.csv"
)
FF1_ABSOLUTE_PAIRWISE_FILE = (
    RESULTS_DIR / "research_question1" / "rq1_absolute_pairwise_dimension_tests.csv"
)
FF2_DISTANCE_PAIRWISE_FILE = (
    RESULTS_DIR / "research_question2" / "rq2_distance_pairwise_feature_tests.csv"
)
FF2_ABSOLUTE_PAIRWISE_FILE = (
    RESULTS_DIR / "research_question2" / "rq2_absolute_pairwise_feature_tests.csv"
)


# ============================================================
# 2. APA-ÄHNLICHE GRUNDEINSTELLUNGEN
# ============================================================

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.unicode_minus": True,
    }
)

PARTY_ORDER = ["cdu/csu", "spd", "fdp", "grüne", "linke", "afd"]

PARTY_LABELS = {
    "cdu/csu": "CDU/CSU",
    "spd": "SPD",
    "fdp": "FDP",
    "grüne": "Bündnis 90/Die Grünen",
    "linke": "Die Linke",
    "afd": "AfD",
}

PARTY_COLORS = {
    "cdu/csu": "#000000",
    "spd": "#E3000F",
    "fdp": "#D4B900",
    "grüne": "#64A12D",
    "linke": "#BE3075",
    "afd": "#009EE0",
}


# ============================================================
# 3. ALLGEMEINE HILFSFUNKTIONEN
# ============================================================

def check_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden:\n{file_path.resolve()}")


def check_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    file_description: str,
) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"In {file_description} fehlen folgende Spalten:\n"
            + "\n".join(f"- {column}" for column in missing)
        )


def remove_chart_junk(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_facecolor("white")


def save_figure(fig: plt.Figure, filename_stem: str) -> None:
    """Speichert ausschließlich eine PNG-Datei mit 300 dpi."""
    png_file = FIGURE_DIR / f"{filename_stem}.png"
    fig.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Gespeichert: {png_file}")


# ============================================================
# 4. H1a/H2a: INTERAKTIONSPLOTS MIT ZWEI PANELS
# ============================================================

def prepare_interaction_data(
    file_path: Path,
    prediction_column: str,
) -> tuple[pd.DataFrame, str, str, str]:
    """Liest und bereinigt eine Vorhersagedatei für einen Interaktionsplot."""
    check_file(file_path)
    df = pd.read_csv(file_path)

    x_column = "afd_support_t"
    group_column = "afd_proximity_ordinal"
    check_columns(
        df,
        [x_column, group_column, prediction_column],
        f"der Vorhersagedatei {file_path}",
    )

    numeric_columns = [x_column, group_column, prediction_column]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=numeric_columns).copy()
    if df.empty:
        raise ValueError(
            f"Keine gültigen Vorhersagewerte in:\n{file_path}\n\n"
            f"Verwendete Spalten:\n"
            f"- AfD-Unterstützung: {x_column}\n"
            f"- Moderator: {group_column}\n"
            f"- Vorhersage: {prediction_column}"
        )

    print("\nEingelesene Interaktionsdaten:")
    print(f"Datei: {file_path}")
    print(f"x-Spalte: {x_column}")
    print(f"Gruppenspalte: {group_column}")
    print(f"Vorhersagespalte: {prediction_column}")
    print(f"Gültige Zeilen: {len(df)}")

    return df, x_column, group_column, prediction_column


def draw_interaction_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_column: str,
    group_column: str,
    prediction_column: str,
    panel_title: str,
    y_label: str,
) -> None:
    party_labels = {
        0: "Die Linke",
        1: "Bündnis 90/Die Grünen",
        2: "SPD",
        3: "FDP",
        4: "CDU/CSU",
    }
    line_styles = {
        0: "-",
        1: "--",
        2: "-.",
        3: ":",
        4: (0, (5, 2)),
    }
    markers = {0: "o", 1: "s", 2: "^", 3: "D", 4: "v"}

    for value in sorted(df[group_column].unique()):
        group = df.loc[df[group_column] == value].sort_values(x_column)
        value_int = int(value)
        marker_positions = np.unique(
            np.linspace(0, len(group) - 1, min(6, len(group)), dtype=int)
        )
        ax.plot(
            group[x_column],
            group[prediction_column],
            label=party_labels.get(value_int, f"Nähe = {value_int}"),
            linestyle=line_styles.get(value_int, "-"),
            marker=markers.get(value_int, "o"),
            markevery=marker_positions,
            linewidth=1.3,
            markersize=4,
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
        )

    ax.set_title(panel_title, fontsize=10, pad=8)
    ax.set_xlabel("AfD-Unterstützung in Prozentpunkten")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", linewidth=0.5, color="0.88")
    remove_chart_junk(ax)


def create_paired_interaction_plot(
    distance_file: Path,
    absolute_file: Path,
    distance_prediction_column: str,
    absolute_prediction_column: str,
    distance_y_label: str,
    absolute_y_label: str,
    filename_stem: str,
) -> None:
    distance = prepare_interaction_data(
        distance_file,
        prediction_column=distance_prediction_column,
    )
    absolute = prepare_interaction_data(
        absolute_file,
        prediction_column=absolute_prediction_column,
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11.2, 4.8),
        sharex=True,
    )

    draw_interaction_panel(
        axes[0], *distance, "Distanzwerte", distance_y_label
    )
    draw_interaction_panel(
        axes[1], *absolute, "Absolute Werte", absolute_y_label
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=min(5, len(labels)),
        handlelength=2.8,
        columnspacing=1.3,
    )

    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=2.5)
    save_figure(fig, filename_stem)
    plt.close(fig)


def create_h1a_interaction_plot() -> None:
    create_paired_interaction_plot(
        H1A_DISTANCE_PREDICTION_FILE,
        H1A_ABSOLUTE_PREDICTION_FILE,
        distance_prediction_column="predicted_distance",
        absolute_prediction_column="predicted_gal_tan",
        distance_y_label="Vorhergesagte GAL–TAN-Distanz zur AfD",
        absolute_y_label="Vorhergesagter absoluter GAL–TAN-Gesamtscore",
        filename_stem="figure_h1a_interaction",
    )


def create_h2a_interaction_plot() -> None:
    create_paired_interaction_plot(
        H2A_DISTANCE_PREDICTION_FILE,
        H2A_ABSOLUTE_PREDICTION_FILE,
        distance_prediction_column="predicted_distance",
        absolute_prediction_column="predicted_populism",
        distance_y_label="Vorhergesagte Populismus-Distanz zur AfD",
        absolute_y_label="Vorhergesagter absoluter Populismus-Gesamtscore",
        filename_stem="figure_h2a_interaction",
    )


# ============================================================
# 5. FF1/FF2: PAARWEISE MATRIZEN MIT ZWEI PANELS
# ============================================================

def load_pairwise_matrix_data(
    file_path: Path,
    label_a_column: str,
    label_b_column: str,
    preferred_order: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Liest die paarweisen Vergleiche ein und erzeugt eine symmetrische
    Matrix der Koeffizientendifferenzen.

    Für FF1 werden die Spalten dimension_a_label und
    dimension_b_label verwendet. Für FF2 werden feature_a_label und
    feature_b_label verwendet.
    """
    check_file(file_path)
    df = pd.read_csv(file_path)

    difference_column = "slope_difference"
    p_column = "p_value_holm"
    wcr31_column = "wcr31_p_value_holm"
    check_columns(
        df,
        [label_a_column, label_b_column, difference_column, p_column, wcr31_column],
        f"der paarweisen Vergleichsdatei {file_path}",
    )

    numeric_columns = [
        difference_column,
        p_column,
        wcr31_column,
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            label_a_column,
            label_b_column,
            difference_column,
        ]
    ).copy()

    labels_in_data = list(
        dict.fromkeys(
            df[label_a_column].astype(str).tolist()
            + df[label_b_column].astype(str).tolist()
        )
    )

    if preferred_order:
        order = [
            label
            for label in preferred_order
            if label in labels_in_data
        ]
        order += [
            label
            for label in labels_in_data
            if label not in order
        ]
    else:
        order = labels_in_data

    if not order:
        raise ValueError(
            f"Keine gültigen Merkmalsbezeichnungen in:\n{file_path}"
        )

    n = len(order)
    lookup = {
        label: index
        for index, label in enumerate(order)
    }

    values = np.full((n, n), np.nan)
    labels = np.full((n, n), "", dtype=object)

    for index in range(n):
        values[index, index] = 0.0
        labels[index, index] = "—"

    for _, row in df.iterrows():
        label_a = str(row[label_a_column])
        label_b = str(row[label_b_column])

        if label_a not in lookup or label_b not in lookup:
            continue

        difference = float(row[difference_column])
        i = lookup[label_a]
        j = lookup[label_b]

        significance_marker = ""

        if (
            pd.notna(row[p_column])
            and float(row[p_column]) < 0.05
        ):
            significance_marker += "*"

        if (
            pd.notna(row[wcr31_column])
            and float(row[wcr31_column]) < 0.05
        ):
            significance_marker += "†"

        values[i, j] = difference
        values[j, i] = -difference

        labels[i, j] = (
            f"{difference:+.3f}{significance_marker}"
        )
        labels[j, i] = (
            f"{-difference:+.3f}{significance_marker}"
        )

    print("\nEingelesene Paarvergleichsdaten:")
    print(f"Datei: {file_path}")
    print(f"Merkmal A: {label_a_column}")
    print(f"Merkmal B: {label_b_column}")
    print(f"Differenz: {difference_column}")
    print(f"Reihenfolge: {order}")

    return values, labels, order


def draw_pairwise_matrix_panel(
    ax: plt.Axes,
    values: np.ndarray,
    labels: np.ndarray,
    dimensions: list[str],
    panel_title: str,
    vmax: float,
    x_axis_label: str,
    y_axis_label: str,
) -> mpl.image.AxesImage:
    shading = np.abs(values)

    image = ax.imshow(
        shading,
        cmap="Greys",
        vmin=0,
        vmax=vmax,
        aspect="equal",
    )

    n = len(dimensions)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(
        dimensions,
        rotation=40,
        ha="right",
    )
    ax.set_yticklabels(dimensions)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title(panel_title, fontsize=10, pad=8)

    for i in range(n):
        for j in range(n):
            text = labels[i, j]

            if text == "":
                continue

            cell_value = shading[i, j]
            text_color = (
                "white"
                if np.isfinite(cell_value)
                and cell_value > vmax * 0.55
                else "black"
            )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8.2,
                color=text_color,
            )

    ax.set_xticks(
        np.arange(-0.5, n, 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, n, 1),
        minor=True,
    )
    ax.grid(
        which="minor",
        linewidth=0.7,
        color="white",
    )
    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    return image


def create_paired_pairwise_matrix(
    distance_file: Path,
    absolute_file: Path,
    label_a_column: str,
    label_b_column: str,
    preferred_order: list[str] | None,
    x_axis_label: str,
    y_axis_label: str,
    filename_stem: str,
) -> None:

    (
        distance_values,
        distance_labels,
        distance_order,
    ) = load_pairwise_matrix_data(
        file_path=distance_file,
        label_a_column=label_a_column,
        label_b_column=label_b_column,
        preferred_order=preferred_order,
    )

    (
        absolute_values,
        absolute_labels,
        absolute_order,
    ) = load_pairwise_matrix_data(
        file_path=absolute_file,
        label_a_column=label_a_column,
        label_b_column=label_b_column,
        preferred_order=preferred_order,
    )

    if distance_order != absolute_order:
        raise ValueError(
            "Die Merkmale oder ihre Reihenfolge unterscheiden sich "
            "zwischen Distanz- und Absolutwertdatei.\n"
            f"Distanz: {distance_order}\n"
            f"Absolut: {absolute_order}"
        )

    maxima = [
        np.nanmax(np.abs(distance_values)),
        np.nanmax(np.abs(absolute_values)),
    ]
    vmax = max(maxima)

    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    n_dimensions = len(distance_order)
    width = max(12.6, 2.0 * n_dimensions + 3.4)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(width, 5.8),
    )

    image = draw_pairwise_matrix_panel(
        ax=axes[0],
        values=distance_values,
        labels=distance_labels,
        dimensions=distance_order,
        panel_title="Distanzwerte",
        vmax=vmax,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )

    draw_pairwise_matrix_panel(
        ax=axes[1],
        values=absolute_values,
        labels=absolute_labels,
        dimensions=absolute_order,
        panel_title="Absolute Werte",
        vmax=vmax,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )

    # Mehr Platz rechts reservieren, damit die Farbskalen-Legende
    # nicht in das rechte Panel hineinragt.
    fig.subplots_adjust(
        left=0.08,
        right=0.84,
        bottom=0.19,
        top=0.90,
        wspace=0.34,
    )

    cbar_ax = fig.add_axes([0.87, 0.24, 0.018, 0.54])
    colorbar = fig.colorbar(
        image,
        cax=cbar_ax,
    )
    colorbar.set_label(
        "Absolute Differenz der\nRegressionskoeffizienten",
        fontsize=9,
        labelpad=10,
    )
    colorbar.ax.tick_params(labelsize=8)

    save_figure(fig, filename_stem)
    plt.close(fig)


def create_ff1_pairwise_matrix() -> None:
    """FF1: Vergleich der fünf GAL–TAN-Subdimensionen."""
    create_paired_pairwise_matrix(
        distance_file=FF1_DISTANCE_PAIRWISE_FILE,
        absolute_file=FF1_ABSOLUTE_PAIRWISE_FILE,
        label_a_column="dimension_a_label",
        label_b_column="dimension_b_label",
        preferred_order=[
            "Umwelt",
            "Migration",
            "Gesellschaftspolitik",
            "Bürgerrechte",
            "Kosmopolitismus",
        ],
        x_axis_label="Subdimension B",
        y_axis_label="Subdimension A",
        filename_stem="figure_ff1_pairwise_matrix",
    )


def create_ff2_pairwise_matrix() -> None:
    """FF2: Vergleich der drei Merkmale populistischer Kommunikation."""
    create_paired_pairwise_matrix(
        distance_file=FF2_DISTANCE_PAIRWISE_FILE,
        absolute_file=FF2_ABSOLUTE_PAIRWISE_FILE,
        label_a_column="feature_a_label",
        label_b_column="feature_b_label",
        preferred_order=[
            "Volkzentrierung",
            "Anti-Elitismus",
            "Ausschluss von Outgroups",
        ],
        x_axis_label="Merkmal B",
        y_axis_label="Merkmal A",
        filename_stem="figure_ff2_pairwise_matrix",
    )


# ============================================================
# 6. ZEITVERLÄUFE: ABSOLUTE SCORES + AFD-UMFRAGEWERTE
# ============================================================

def create_score_and_afd_poll_plot(
    outcome: str,
    y_label: str,
    filename_stem: str,
) -> None:
    check_file(PARTY_INTERVAL_FILE)
    df = pd.read_csv(PARTY_INTERVAL_FILE)

    afd_poll_column = "afd_support_t"

    required_columns = ["poll_date", "party_clean", outcome, afd_poll_column]
    check_columns(df, required_columns, "dem Partei–Umfrageintervall-Datensatz")

    plot_df = df[required_columns].copy()
    plot_df["poll_date"] = pd.to_datetime(plot_df["poll_date"], errors="coerce")
    plot_df[outcome] = pd.to_numeric(plot_df[outcome], errors="coerce")
    plot_df[afd_poll_column] = pd.to_numeric(plot_df[afd_poll_column], errors="coerce")
    plot_df["party_clean"] = (
        plot_df["party_clean"].astype(str).str.strip().str.lower()
    )
    plot_df = plot_df.loc[plot_df["party_clean"].isin(PARTY_ORDER)].copy()
    plot_df = plot_df.dropna(subset=["poll_date", "party_clean", outcome])

    available_parties = set(plot_df["party_clean"].unique())
    missing_parties = [party for party in PARTY_ORDER if party not in available_parties]
    if missing_parties:
        raise ValueError(
            f"Für {outcome} fehlen folgende Parteien:\n"
            + "\n".join(f"- {PARTY_LABELS[party]}" for party in missing_parties)
        )

    score_df = (
        plot_df.groupby(["poll_date", "party_clean"], as_index=False, observed=True)[outcome]
        .mean()
        .sort_values(["party_clean", "poll_date"])
    )

    afd_poll_df = (
        plot_df.loc[plot_df[afd_poll_column].notna(), ["poll_date", afd_poll_column]]
        .groupby("poll_date", as_index=False)[afd_poll_column]
        .mean()
        .sort_values("poll_date")
    )
    if afd_poll_df.empty:
        raise ValueError("Keine gültigen AfD-Umfragewerte vorhanden.")

    fig, score_ax = plt.subplots(figsize=(8.4, 5.1))

    for party in PARTY_ORDER:
        party_df = score_df.loc[score_df["party_clean"] == party].sort_values("poll_date")
        score_ax.plot(
            party_df["poll_date"],
            party_df[outcome],
            label=PARTY_LABELS[party],
            color=PARTY_COLORS[party],
            linewidth=1.35,
            linestyle="-",
            zorder=2,
        )

    score_ax.set_xlabel("Datum der Umfrage")
    score_ax.set_ylabel(y_label)
    score_ax.grid(axis="y", linewidth=0.5, color="0.88", zorder=0)
    score_ax.grid(axis="x", visible=False)
    remove_chart_junk(score_ax)
    score_ax.xaxis.set_major_locator(mdates.YearLocator())
    score_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    poll_ax = score_ax.twinx()
    poll_ax.plot(
        afd_poll_df["poll_date"],
        afd_poll_df[afd_poll_column],
        label="AfD-Umfragewert",
        color="#444444",
        linestyle="--",
        linewidth=2.0,
        zorder=4,
    )
    poll_ax.set_ylabel("AfD-Umfragewert in Prozentpunkten")
    poll_ax.spines["top"].set_visible(False)
    poll_ax.spines["right"].set_linewidth(0.8)

    score_handles, score_labels = score_ax.get_legend_handles_labels()
    poll_handles, poll_labels = poll_ax.get_legend_handles_labels()
    score_ax.legend(
        score_handles + poll_handles,
        score_labels + poll_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        handlelength=2.8,
        columnspacing=1.3,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.27, right=0.88)
    save_figure(fig, filename_stem)
    plt.close(fig)


def create_gal_tan_and_afd_poll_plot() -> None:
    create_score_and_afd_poll_plot(
        outcome="mean_gal_tan_score",
        y_label="Mittlerer GAL–TAN-Gesamtscore",
        filename_stem="figure_gal_tan_scores_and_afd_poll_over_time",
    )


def create_populism_and_afd_poll_plot() -> None:
    create_score_and_afd_poll_plot(
        outcome="mean_populism_score",
        y_label="Mittlerer Populismus-Gesamtscore",
        filename_stem="figure_populism_scores_and_afd_poll_over_time",
    )


# ============================================================
# 7. ALLE GEWÜNSCHTEN ABBILDUNGEN ERSTELLEN
# ============================================================

def main() -> None:
    print("=" * 72)
    print("Erstellung der ausgewählten APA-7-Abbildungen")
    print("=" * 72)

    # H1a/H2a: jeweils Distanzwerte links und absolute Werte rechts
    create_h1a_interaction_plot()
    create_h2a_interaction_plot()

    # FF1/FF2: jeweils Distanzwerte links und absolute Werte rechts
    create_ff1_pairwise_matrix()
    create_ff2_pairwise_matrix()

    # Absolute Gesamtscores mit AfD-Umfragewerten
    create_gal_tan_and_afd_poll_plot()
    create_populism_and_afd_poll_plot()

    print("\nAlle gewünschten Abbildungen wurden erstellt.")
    print(f"Ausgabeordner: {FIGURE_DIR.resolve()}")


if __name__ == "__main__":
    main()