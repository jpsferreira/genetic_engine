"""Biologically-inspired interactive visualization of genetic algorithm evolution.

Creates a rich, animated Plotly dashboard — *The Evolution Chronicle* — that
tells the story of natural selection through four coordinated panels:

1. **The Ecosystem** — animated scatter of organisms in gene space
2. **The Genome**    — heatmap showing how the fittest DNA converges
3. **The Chronicle** — fitness curves over generations
4. **Natural Selection** — every organism's fate across time

Usage::

    from genetic_opt.sga.utils.bio_visualization import create_evolution_chronicle

    create_evolution_chronicle(
        "results/.../population.csv",
        fitness_function=my_func,       # optional but recommended
        metrics_file="results/.../metrics.csv",  # optional
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError(
        "Plotly is required for bio_visualization. "
        "Install it with: pip install plotly"
    )

# ── Theme ─────────────────────────────────────────────────────────────────

_BG          = "#0d1117"
_PANEL_BG    = "#161b22"
_GRID        = "rgba(48,54,61,0.4)"
_TEXT         = "#c9d1d9"
_TEXT_MUTED   = "#8b949e"
_ACCENT       = "#58a6ff"
_GREEN        = "#3fb950"
_YELLOW       = "#d29922"
_RED          = "#f85149"
_GOLD         = "#ffd700"

# Fitness colorscale: red (low) → amber → green (high)
_FITNESS_SCALE = [
    [0.0,  _RED],
    [0.35, "#f0883e"],
    [0.55, _YELLOW],
    [0.8,  "#56d364"],
    [1.0,  _GREEN],
]

_GENOME_SCALE = [
    [0.0,  "#0d1b2a"],
    [0.2,  "#1b2838"],
    [0.4,  "#2a4a6b"],
    [0.6,  "#3a7ca5"],
    [0.8,  "#58a6ff"],
    [1.0,  "#a5d6ff"],
]

_ERAS = [
    (0.00, "Dawn of Life"),
    (0.12, "Cambrian Explosion"),
    (0.30, "Age of Exploration"),
    (0.55, "Natural Selection Intensifies"),
    (0.75, "Survival of the Fittest"),
    (0.90, "Convergent Evolution"),
]

MAX_ANIM_FRAMES = 60


# ── Data preparation ──────────────────────────────────────────────────────

def _load_and_prepare(
    population_file: str,
    metrics_file: Optional[str],
    fitness_function: Optional[Callable],
) -> Dict:
    """Load CSVs, compute fitness & projections, return a data bundle."""
    pop_df = pd.read_csv(population_file)
    gene_cols = [c for c in pop_df.columns if c.startswith("gene_")]
    n_genes = len(gene_cols)
    generations = sorted(pop_df["generation"].unique())
    n_gens = len(generations)

    # ── Fitness ───────────────────────────────────────────────────
    has_fitness = False
    if fitness_function is not None:
        pop_df["fitness"] = pop_df[gene_cols].apply(
            lambda r: fitness_function(r.tolist()), axis=1,
        )
        has_fitness = True

    # ── Metrics ───────────────────────────────────────────────────
    metrics_df = None
    if metrics_file and Path(metrics_file).exists():
        metrics_df = pd.read_csv(metrics_file)

    # ── 2-D projection ───────────────────────────────────────────
    if n_genes == 1:
        pop_df["x"] = pop_df[gene_cols[0]]
        pop_df["y"] = 0.0
        x_label, y_label = "Gene 1", ""
    elif n_genes == 2:
        pop_df["x"] = pop_df[gene_cols[0]]
        pop_df["y"] = pop_df[gene_cols[1]]
        x_label, y_label = "Gene 1", "Gene 2"
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        proj = pca.fit_transform(pop_df[gene_cols].values)
        pop_df["x"] = proj[:, 0]
        pop_df["y"] = proj[:, 1]
        ev = pca.explained_variance_ratio_
        x_label = f"PC 1 ({ev[0]:.0%} var)"
        y_label = f"PC 2 ({ev[1]:.0%} var)"

    # ── Per-generation statistics ─────────────────────────────────
    gen_stats: List[Dict] = []
    for gen in generations:
        gd = pop_df[pop_df["generation"] == gen]
        entry: Dict = {"generation": gen}
        entry["mean_genes"] = [gd[c].mean() for c in gene_cols]
        entry["std_genes"] = [gd[c].std() for c in gene_cols]
        if has_fitness:
            best_idx = gd["fitness"].idxmax()
            entry["best_fitness"] = gd.loc[best_idx, "fitness"]
            entry["avg_fitness"] = gd["fitness"].mean()
            entry["worst_fitness"] = gd["fitness"].min()
            entry["std_fitness"] = gd["fitness"].std()
            entry["best_genes"] = [gd.loc[best_idx, c] for c in gene_cols]
            entry["best_x"] = gd.loc[best_idx, "x"]
            entry["best_y"] = gd.loc[best_idx, "y"]
        gen_stats.append(entry)

    # ── Sample generations for animation ──────────────────────────
    if n_gens <= MAX_ANIM_FRAMES:
        sampled = generations
    else:
        idx = np.linspace(0, n_gens - 1, MAX_ANIM_FRAMES).astype(int)
        sampled = [generations[i] for i in idx]

    return dict(
        pop_df=pop_df,
        gene_cols=gene_cols,
        n_genes=n_genes,
        generations=generations,
        n_gens=n_gens,
        has_fitness=has_fitness,
        metrics_df=metrics_df,
        gen_stats=gen_stats,
        sampled=sampled,
        x_label=x_label,
        y_label=y_label,
    )


def _era_label(gen: int, total: int) -> str:
    progress = gen / max(total - 1, 1)
    label = _ERAS[0][1]
    for threshold, name in _ERAS:
        if progress >= threshold:
            label = name
    return label


def _norm(values, vmin, vmax):
    """Normalise *values* to [0, 1]."""
    r = vmax - vmin
    if r < 1e-12:
        return [0.5] * len(values)
    return [(v - vmin) / r for v in values]


# ── Panel builders ────────────────────────────────────────────────────────

def _build_ecosystem(fig, data, row, col):
    """Add ecosystem traces (ghost, population, best, label) and return frames."""
    pop_df = data["pop_df"]
    sampled = data["sampled"]
    has_fitness = data["has_fitness"]
    gen_stats = {s["generation"]: s for s in data["gen_stats"]}
    n_gens = data["n_gens"]
    gens = data["generations"]

    # Fitness range (for consistent colouring)
    if has_fitness:
        f_min = pop_df["fitness"].min()
        f_max = pop_df["fitness"].max()

    def _traces_for(gen_idx: int):
        gen = sampled[gen_idx]
        prev_gen = sampled[max(0, gen_idx - 1)]
        gd = pop_df[pop_df["generation"] == gen]
        pd_ = pop_df[pop_df["generation"] == prev_gen]
        stats = gen_stats[gen]
        era = _era_label(gen, gens[-1] + 1)

        if has_fitness:
            f_norm = _norm(gd["fitness"].tolist(), f_min, f_max)
            f_norm_prev = _norm(pd_["fitness"].tolist(), f_min, f_max)
            sizes = [max(4, 5 + 14 * fn) for fn in f_norm]
            hover = [
                f"<b>Organism #{i}</b><br>"
                f"Fitness: {f:.4f}<br>"
                f"Rank: {'Elite' if fn > 0.9 else 'Survivor' if fn > 0.5 else 'Struggling'}"
                for i, (f, fn) in enumerate(zip(gd["fitness"], f_norm))
            ]
            color = gd["fitness"].tolist()
            color_prev = pd_["fitness"].tolist()
        else:
            sizes = [7] * len(gd)
            hover = [f"Organism #{i}" for i in range(len(gd))]
            color = [_ACCENT] * len(gd)
            color_prev = [_ACCENT] * len(pd_)
            f_norm = None

        # Ghost (previous generation)
        ghost = go.Scatter(
            x=pd_["x"].tolist(), y=pd_["y"].tolist(),
            mode="markers",
            marker=dict(
                size=5, opacity=0.15,
                color=color_prev,
                **(dict(colorscale=_FITNESS_SCALE, cmin=f_min, cmax=f_max)
                   if has_fitness else {}),
            ),
            hoverinfo="skip",
            showlegend=False,
        )

        # Current population
        pop_trace = go.Scatter(
            x=gd["x"].tolist(), y=gd["y"].tolist(),
            mode="markers",
            marker=dict(
                size=sizes, opacity=0.85,
                color=color,
                **(dict(colorscale=_FITNESS_SCALE, cmin=f_min, cmax=f_max,
                        colorbar=dict(
                            title=dict(text="Fitness", font=dict(color=_TEXT)),
                            tickfont=dict(color=_TEXT_MUTED),
                            bgcolor=_PANEL_BG, borderwidth=0,
                            len=0.35, y=0.82,
                        )) if has_fitness else {}),
                line=dict(width=0.5, color="rgba(255,255,255,0.4)"),
            ),
            customdata=list(range(len(gd))),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )

        # Best individual
        if has_fitness:
            best_trace = go.Scatter(
                x=[stats["best_x"]], y=[stats["best_y"]],
                mode="markers+text",
                marker=dict(size=22, symbol="star", color=_GOLD,
                            line=dict(width=2, color="white")),
                text=["BEST"],
                textposition="top center",
                textfont=dict(color=_GOLD, size=10, family="Arial Black"),
                hovertext=f"<b>Fittest Organism</b><br>Fitness: {stats['best_fitness']:.6f}",
                hoverinfo="text",
                showlegend=False,
            )
        else:
            best_trace = go.Scatter(
                x=[], y=[], mode="markers", showlegend=False, hoverinfo="skip",
            )

        # Era label (rendered as a text trace)
        best_f_text = f"   |   Best: {stats['best_fitness']:.4f}" if has_fitness else ""
        label_trace = go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=0, opacity=0),
            showlegend=False, hoverinfo="skip",
        )

        return ghost, pop_trace, best_trace, label_trace, era, gen, best_f_text

    # Initial traces (generation 0)
    ghost0, pop0, best0, lbl0, era0, gen0, bf0 = _traces_for(0)
    for t in [ghost0, pop0, best0, lbl0]:
        fig.add_trace(t, row=row, col=col)

    # Record base trace indices (needed for frames)
    base = len(fig.data) - 4

    # Build frames
    frames = []
    for gi in range(len(sampled)):
        ghost, pop_t, best_t, lbl_t, era, gen, bf_text = _traces_for(gi)
        frames.append(go.Frame(
            data=[ghost, pop_t, best_t, lbl_t],
            traces=[base, base + 1, base + 2, base + 3],
            name=str(sampled[gi]),
            layout=dict(
                annotations=[dict(
                    x=0.01, y=0.99, xref="paper", yref="paper",
                    xanchor="left", yanchor="top",
                    text=(
                        f"<b>Generation {gen}</b>   |   {era}{bf_text}"
                    ),
                    font=dict(size=14, color=_TEXT, family="Arial"),
                    bgcolor="rgba(22,27,34,0.85)",
                    bordercolor=_ACCENT, borderwidth=1, borderpad=6,
                    showarrow=False,
                )],
            ),
        ))

    return frames


def _build_genome(fig, data, row, col):
    """Heatmap of the fittest individual's DNA over generations."""
    gen_stats = data["gen_stats"]
    gene_cols = data["gene_cols"]
    has_fitness = data["has_fitness"]

    if has_fitness:
        # Best individual's genes per generation
        z = np.array([s["best_genes"] for s in gen_stats])
        title_extra = " (Fittest Individual)"
    else:
        # Population mean genes per generation
        z = np.array([s["mean_genes"] for s in gen_stats])
        title_extra = " (Population Mean)"

    # Normalise per gene for consistent coloring
    z_norm = np.zeros_like(z)
    for j in range(z.shape[1]):
        col_vals = z[:, j]
        vmin, vmax = col_vals.min(), col_vals.max()
        if vmax - vmin > 1e-12:
            z_norm[:, j] = (col_vals - vmin) / (vmax - vmin)
        else:
            z_norm[:, j] = 0.5

    gene_labels = [f"G{i+1}" for i in range(len(gene_cols))]
    gen_labels = [str(s["generation"]) for s in gen_stats]

    fig.add_trace(go.Heatmap(
        z=z_norm,
        x=gene_labels,
        y=gen_labels,
        colorscale=_GENOME_SCALE,
        customdata=z,
        hovertemplate=(
            "Generation %{y}<br>"
            "%{x}: %{customdata:.4f}<br>"
            "<extra></extra>"
        ),
        showscale=False,
    ), row=row, col=col)

    fig.update_yaxes(
        title_text="Generation", autorange="reversed",
        row=row, col=col,
    )
    fig.update_xaxes(title_text="Gene", row=row, col=col)


def _build_chronicle(fig, data, row, col):
    """Fitness curves: best, average, and population band."""
    gen_stats = data["gen_stats"]
    has_fitness = data["has_fitness"]
    metrics_df = data["metrics_df"]

    gens_x, best_y, avg_y, worst_y = [], [], [], []

    if has_fitness:
        for s in gen_stats:
            gens_x.append(s["generation"])
            best_y.append(s["best_fitness"])
            avg_y.append(s["avg_fitness"])
            worst_y.append(s["worst_fitness"])
    elif metrics_df is not None and "best_fitness" in metrics_df.columns:
        gens_x = metrics_df["generation"].tolist()
        best_y = metrics_df["best_fitness"].tolist()
        if "avg_fitness" in metrics_df.columns:
            avg_y = metrics_df["avg_fitness"].tolist()
        if "std_fitness" in metrics_df.columns:
            avg_vals = metrics_df.get("avg_fitness", metrics_df["best_fitness"])
            std_vals = metrics_df["std_fitness"]
            worst_y = (avg_vals - std_vals).tolist()
    else:
        # Fallback: gene diversity (sum of gene std devs)
        for s in gen_stats:
            gens_x.append(s["generation"])
            best_y.append(sum(s["std_genes"]))
        fig.add_trace(go.Scatter(
            x=gens_x, y=best_y,
            mode="lines",
            line=dict(color=_ACCENT, width=2),
            name="Gene Diversity",
            hovertemplate="Gen %{x}<br>Diversity: %{y:.4f}<extra></extra>",
            showlegend=False,
        ), row=row, col=col)
        fig.update_yaxes(title_text="Gene Diversity", row=row, col=col)
        fig.update_xaxes(title_text="Generation", row=row, col=col)
        return

    # Band (worst to best)
    if worst_y:
        fig.add_trace(go.Scatter(
            x=gens_x + gens_x[::-1],
            y=best_y + worst_y[::-1],
            fill="toself",
            fillcolor="rgba(88,166,255,0.08)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ), row=row, col=col)

    # Average
    if avg_y:
        fig.add_trace(go.Scatter(
            x=gens_x, y=avg_y,
            mode="lines",
            line=dict(color=_TEXT_MUTED, width=1.5, dash="dot"),
            name="Average",
            hovertemplate="Gen %{x}<br>Avg: %{y:.4f}<extra></extra>",
            showlegend=False,
        ), row=row, col=col)

    # Best
    fig.add_trace(go.Scatter(
        x=gens_x, y=best_y,
        mode="lines",
        line=dict(color=_GREEN, width=2.5),
        name="Best Fitness",
        hovertemplate="Gen %{x}<br>Best: %{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=row, col=col)

    fig.update_yaxes(title_text="Fitness", row=row, col=col)
    fig.update_xaxes(title_text="Generation", row=row, col=col)


def _build_selection(fig, data, row, col):
    """Scatter of every organism's fitness — the selection funnel."""
    pop_df = data["pop_df"]
    has_fitness = data["has_fitness"]
    gene_cols = data["gene_cols"]

    if not has_fitness:
        # Fallback: show per-gene std over time (diversity decay)
        gen_stats = data["gen_stats"]
        gens = [s["generation"] for s in gen_stats]
        for gi, gc in enumerate(gene_cols):
            stds = [s["std_genes"][gi] for s in gen_stats]
            fig.add_trace(go.Scatter(
                x=gens, y=stds,
                mode="lines",
                line=dict(width=1.5),
                name=f"Gene {gi+1}",
                hovertemplate=f"Gene {gi+1}<br>Gen %{{x}}<br>Std: %{{y:.4f}}<extra></extra>",
                showlegend=(gi < 8),
            ), row=row, col=col)
        fig.update_yaxes(title_text="Gene Std Dev (Diversity)", row=row, col=col)
        fig.update_xaxes(title_text="Generation", row=row, col=col)
        return

    # Subsample for performance (cap at ~15 000 points)
    max_points = 15_000
    total = len(pop_df)
    if total > max_points:
        frac = max_points / total
        plot_df = pop_df.groupby("generation", group_keys=False).apply(
            lambda g: g.sample(frac=min(1.0, frac), random_state=0)
        )
    else:
        plot_df = pop_df

    f_min = pop_df["fitness"].min()
    f_max = pop_df["fitness"].max()

    # Percentile rank within each generation
    def _rank_label(row):
        gd = pop_df[pop_df["generation"] == row["generation"]]
        rank = (gd["fitness"] <= row["fitness"]).mean()
        if rank >= 0.90:
            return "Elite"
        elif rank >= 0.50:
            return "Survivor"
        else:
            return "Struggling"

    fig.add_trace(go.Scatter(
        x=plot_df["generation"].tolist(),
        y=plot_df["fitness"].tolist(),
        mode="markers",
        marker=dict(
            size=3,
            opacity=0.5,
            color=plot_df["fitness"].tolist(),
            colorscale=_FITNESS_SCALE,
            cmin=f_min, cmax=f_max,
        ),
        hovertemplate="Gen %{x}<br>Fitness: %{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=row, col=col)

    # Overlay best fitness line
    gen_stats = data["gen_stats"]
    fig.add_trace(go.Scatter(
        x=[s["generation"] for s in gen_stats],
        y=[s["best_fitness"] for s in gen_stats],
        mode="lines",
        line=dict(color=_GOLD, width=2),
        hovertemplate="Gen %{x}<br>Best: %{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=row, col=col)

    fig.update_yaxes(title_text="Fitness", row=row, col=col)
    fig.update_xaxes(title_text="Generation", row=row, col=col)


# ── Main entry point ──────────────────────────────────────────────────────

def create_evolution_chronicle(
    population_file: str,
    output_file: str = "evolution_chronicle.html",
    metrics_file: Optional[str] = None,
    fitness_function: Optional[Callable] = None,
    title: str = "The Evolution Chronicle",
    auto_open: bool = False,
) -> str:
    """Create an interactive, biologically-inspired evolution visualisation.

    Generates a self-contained HTML file with an animated dashboard that
    shows how a genetic algorithm population evolves over generations,
    framed as a story of natural selection.

    Args:
        population_file: Path to a population history CSV exported by
            :class:`~genetic_opt.sga.optimizer.GeneticOptimizer`.
        output_file: Where to write the HTML file.
        metrics_file: Optional metrics CSV (adds richer chronicle panel).
        fitness_function: Optional objective function.  When provided,
            fitness is computed for every individual, enabling colour-coded
            organisms, the selection-pressure panel, and richer hover text.
        title: Dashboard title.
        auto_open: Open the HTML file in the default browser.

    Returns:
        Path to the generated HTML file.
    """
    data = _load_and_prepare(population_file, metrics_file, fitness_function)

    has_fitness = data["has_fitness"]

    panel_titles = [
        "The Ecosystem  --  Population in Gene Space",
        "The Genome  --  DNA Across Generations",
        ("The Chronicle  --  Fitness Over Time"
         if has_fitness or data["metrics_df"] is not None
         else "The Chronicle  --  Gene Diversity"),
        ("Natural Selection  --  Every Organism's Fate"
         if has_fitness
         else "Diversity Decay  --  Gene Spread Over Time"),
    ]

    fig = make_subplots(
        rows=3, cols=2,
        row_heights=[0.44, 0.28, 0.28],
        column_widths=[0.48, 0.52],
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{"colspan": 2}, None],
        ],
        subplot_titles=panel_titles,
        vertical_spacing=0.09,
        horizontal_spacing=0.07,
    )

    # ── Build panels ──────────────────────────────────────────────
    frames = _build_ecosystem(fig, data, row=1, col=1)
    _build_genome(fig, data, row=2, col=1)
    _build_chronicle(fig, data, row=2, col=2)
    _build_selection(fig, data, row=3, col=1)

    fig.frames = frames

    # ── Axis labels for ecosystem ─────────────────────────────────
    fig.update_xaxes(title_text=data["x_label"], row=1, col=1)
    fig.update_yaxes(title_text=data["y_label"], row=1, col=1)

    # ── Animation controls ────────────────────────────────────────
    sampled = data["sampled"]
    slider_steps = [
        dict(
            args=[[str(g)], dict(frame=dict(duration=120, redraw=True),
                                 mode="immediate",
                                 transition=dict(duration=60))],
            label=str(g),
            method="animate",
        )
        for g in sampled
    ]

    fig.update_layout(
        sliders=[dict(
            active=0,
            steps=slider_steps,
            currentvalue=dict(
                prefix="Generation: ",
                font=dict(size=13, color=_TEXT),
            ),
            pad=dict(t=40, b=10),
            len=0.92,
            x=0.04,
            bgcolor=_PANEL_BG,
            activebgcolor=_ACCENT,
            bordercolor=_GRID,
            font=dict(color=_TEXT_MUTED, size=9),
        )],
        updatemenus=[dict(
            type="buttons",
            x=0.0, y=-0.04,
            xanchor="left",
            buttons=[
                dict(
                    label="  Play  ",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=120, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=60),
                    )],
                ),
                dict(
                    label="  Pause  ",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                    )],
                ),
            ],
            font=dict(color=_TEXT, size=12),
            bgcolor=_PANEL_BG,
            bordercolor=_GRID,
        )],
    )

    # ── Initial era annotation ────────────────────────────────────
    s0 = data["gen_stats"][0]
    era0 = _era_label(0, data["generations"][-1] + 1)
    bf0 = f"   |   Best: {s0['best_fitness']:.4f}" if has_fitness else ""
    fig.update_layout(
        annotations=[dict(
            x=0.01, y=0.99, xref="paper", yref="paper",
            xanchor="left", yanchor="top",
            text=f"<b>Generation 0</b>   |   {era0}{bf0}",
            font=dict(size=14, color=_TEXT, family="Arial"),
            bgcolor="rgba(22,27,34,0.85)",
            bordercolor=_ACCENT, borderwidth=1, borderpad=6,
            showarrow=False,
        )],
    )

    # ── Global theme ──────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title}</b><br>"
                f"<span style='font-size:13px;color:{_TEXT_MUTED}'>"
                f"{data['n_gens']} generations  |  "
                f"{data['pop_df']['individual_id'].nunique()} organisms per generation  |  "
                f"{data['n_genes']} genes</span>"
            ),
            font=dict(size=22, color=_TEXT, family="Georgia, serif"),
            x=0.5, xanchor="center",
            y=0.98, yanchor="top",
        ),
        paper_bgcolor=_BG,
        plot_bgcolor=_PANEL_BG,
        font=dict(color=_TEXT, family="Arial"),
        margin=dict(l=60, r=30, t=100, b=80),
        height=1050,
    )

    # Subtitle styling
    for ann in fig.layout.annotations:
        if hasattr(ann, "text") and ann.text in panel_titles:
            ann.font = dict(size=13, color=_ACCENT, family="Georgia, serif")

    # Axis styling
    fig.update_xaxes(
        gridcolor=_GRID, zeroline=False,
        title_font=dict(color=_TEXT_MUTED, size=11),
        tickfont=dict(color=_TEXT_MUTED, size=10),
    )
    fig.update_yaxes(
        gridcolor=_GRID, zeroline=False,
        title_font=dict(color=_TEXT_MUTED, size=11),
        tickfont=dict(color=_TEXT_MUTED, size=10),
    )

    # ── Save ──────────────────────────────────────────────────────
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_path),
        include_plotlyjs=True,
        full_html=True,
        auto_open=auto_open,
        config=dict(
            displayModeBar=True,
            modeBarButtonsToRemove=["lasso2d", "select2d"],
            displaylogo=False,
        ),
    )

    return str(output_path)
