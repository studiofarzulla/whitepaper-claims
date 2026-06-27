#!/usr/bin/env python3
"""
Regenerate the two claims-side method-agreement figures:

    1. method_comparison_3way      -- grouped bars, per-category mean
                                      classification score for each of the
                                      three NLP methods (BART-NLI, embedding,
                                      Ministral-3).
    2. category_agreement_heatmap  -- per-category Pearson r for each method
                                      pair (BART vs Embed, BART vs LLM,
                                      Embed vs LLM).

Both are tensor-independent: they come purely from the NLP classification
matrices and the three-way method comparison, so they survive the factor-leg
cut. The original ad-hoc generator was never committed; this script restores a
reproducible path and asserts that what it plots matches the cited
`outputs/nlp/method_agreement.json` (and therefore Table 1 and the per-category
text in the manuscript).

PROVENANCE NOTE. `method_agreement.json` is the three-way comparison across
BART-NLI, embedding and Ministral-3. As of the June-2026 reliability-leg
reconciliation, ALL THREE canonical `outputs/nlp/` matrices are at the
content-verified n = 43 corpus: BART-NLI and the local LLM were rebuilt in the
clean re-run, and the embedding matrix was subsequently regenerated at n = 43
(all-mpnet-base-v2 over the clean-43 extracted chunks; see
`src/nlp/embedding_classifier.py`). `compare_methods.py` was then re-run, so
`method_agreement.json`, Table 1, and these two figures are now the clean-43
three-way comparison. `resolve_matrix_dir` below still guards against a
length mismatch and would fall back to the preserved n = 38 set in
`outputs/_pre_rerun_backup_jun27-nlp/`, but under the reconciled state it
uses the canonical n = 43 matrices directly.

Usage:
    python scripts/plot_method_agreement_figures.py
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

CATEGORIES = [
    "store_of_value", "medium_of_exchange", "smart_contracts", "defi",
    "governance", "scalability", "privacy", "interoperability",
    "data_storage", "oracle",
]

# Matrix file -> method label (mirrors scripts/compare_methods.py)
MATRIX_FILES = {
    "bart_nli": "claims_matrix.npy",
    "embedding": "claims_matrix_embedding.npy",
    "ministral3": "claims_matrix_llm.npy",
}

# Colours and display names matched to the original figures.
METHOD_DISPLAY = {"bart_nli": "BART-NLI", "embedding": "Embedding", "ministral3": "Ministral-3"}
METHOD_COLOUR = {"bart_nli": "#2ecc71", "embedding": "#3498db", "ministral3": "#e74c3c"}

PAIR_LABEL = {
    ("bart_nli", "embedding"): "BART vs\nEmbed",
    ("bart_nli", "ministral3"): "BART vs\nLLM",
    ("embedding", "ministral3"): "Embed vs\nLLM",
}


def load_matrices(matrix_dir: Path) -> dict[str, np.ndarray]:
    return {name: np.load(matrix_dir / fn) for name, fn in MATRIX_FILES.items()}


def resolve_matrix_dir(base: Path) -> Path:
    """Pick a matrix directory in which all three matrices share a length.

    Prefers canonical `outputs/nlp/`; falls back to the pre-re-run backup if the
    canonical matrices are of unequal length (the post-clean-re-run state).
    """
    canonical = base / "outputs" / "nlp"
    backup = base / "outputs" / "_pre_rerun_backup_jun27-nlp"

    shapes = {n: np.load(canonical / fn).shape[0] for n, fn in MATRIX_FILES.items()}
    if len(set(shapes.values())) == 1:
        print(f"[provenance] using canonical outputs/nlp/ (all matrices n={next(iter(shapes.values()))})")
        return canonical

    print(f"[provenance] canonical outputs/nlp/ matrices are of unequal length {shapes};")
    print("[provenance] the three-way comparison is not recomputable there. Falling back to")
    print("[provenance] outputs/_pre_rerun_backup_jun27-nlp/ (the n=38 set behind method_agreement.json).")
    if not backup.exists():
        raise SystemExit("No equal-length matrix set available (backup missing).")
    bshapes = {n: np.load(backup / fn).shape[0] for n, fn in MATRIX_FILES.items()}
    if len(set(bshapes.values())) != 1:
        raise SystemExit(f"Backup matrices also of unequal length {bshapes}.")
    return backup


def verify_against_json(matrices: dict[str, np.ndarray], json_path: Path) -> None:
    """Assert the plotted matrices reproduce the cited method_agreement.json."""
    ref = json.load(open(json_path))

    for (n1, m1), (n2, m2) in combinations(matrices.items(), 2):
        key = f"{n1}_vs_{n2}"
        r, _ = pearsonr(m1.flatten(), m2.flatten())
        rho, _ = spearmanr(m1.flatten(), m2.flatten())
        jr = ref["pairwise_correlations"][key]["pearson"]["r"]
        jrho = ref["pairwise_correlations"][key]["spearman"]["rho"]
        assert abs(r - jr) < 1e-6 and abs(rho - jrho) < 1e-6, (
            f"{key}: plotted ({r:.6f}/{rho:.6f}) != json ({jr:.6f}/{jrho:.6f})"
        )

    for i, cat in enumerate(CATEGORIES):
        pair_corrs = [pearsonr(m1[:, i], m2[:, i])[0]
                      for (n1, m1), (n2, m2) in combinations(matrices.items(), 2)]
        jmean = ref["category_agreement"][cat]["mean_correlation"]
        assert abs(np.mean(pair_corrs) - jmean) < 1e-6, (
            f"{cat}: plotted mean {np.mean(pair_corrs):.6f} != json {jmean:.6f}"
        )
    print(f"[verify] plotted matrices reproduce {json_path.name} exactly (pairwise + per-category means).")


def plot_method_comparison_3way(matrices: dict[str, np.ndarray], out_paths: list[Path]) -> None:
    methods = list(matrices.keys())
    col_means = {m: matrices[m].mean(axis=0) for m in methods}

    x = np.arange(len(CATEGORIES))
    width = 0.26

    fig, ax = plt.subplots(figsize=(14, 6))
    for j, m in enumerate(methods):
        ax.bar(x + (j - 1) * width, col_means[m], width,
               label=METHOD_DISPLAY[m], color=METHOD_COLOUR[m])

    ax.set_ylabel("Mean Score")
    ax.set_title("Multi-Method Classification Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in CATEGORIES])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    for p in out_paths:
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"[saved] {p}")
    plt.close(fig)


def plot_category_agreement_heatmap(matrices: dict[str, np.ndarray], out_paths: list[Path]) -> None:
    pairs = list(combinations(matrices.keys(), 2))
    data = np.zeros((len(CATEGORIES), len(pairs)))
    for i, cat in enumerate(CATEGORIES):
        for j, (n1, n2) in enumerate(pairs):
            data[i, j] = pearsonr(matrices[n1][:, i], matrices[n2][:, i])[0]

    col_labels = [PAIR_LABEL[p] for p in pairs]

    fig, ax = plt.subplots(figsize=(9, 7.2))
    sns.heatmap(
        data, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=-0.5, vmax=1.0,
        xticklabels=col_labels, yticklabels=CATEGORIES,
        cbar_kws={"label": "Pearson r"},
        annot_kws={"fontweight": "bold"}, ax=ax,
    )
    ax.set_title("Per-Category Inter-Method Agreement")
    ax.set_yticklabels(CATEGORIES, rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=0)

    for p in out_paths:
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"[saved] {p}")
    plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parent.parent           # code/
    paper_fig = base.parent / "paper" / "figures"           # whitepaper-claims/paper/figures
    out_fig = base / "outputs" / "figures"

    matrix_dir = resolve_matrix_dir(base)
    matrices = load_matrices(matrix_dir)
    n = next(iter(matrices.values())).shape[0]
    print(f"[load] {matrix_dir} | three methods at n={n}: "
          + ", ".join(f"{k}{v.shape}" for k, v in matrices.items()))

    verify_against_json(matrices, base / "outputs" / "nlp" / "method_agreement.json")

    plot_method_comparison_3way(matrices, [
        out_fig / "method_comparison_3way.png",
        paper_fig / "method_comparison_3way.png",
        paper_fig / "method_comparison_3way.pdf",
    ])
    plot_category_agreement_heatmap(matrices, [
        out_fig / "category_agreement_heatmap.png",
        paper_fig / "category_agreement_heatmap.png",
        paper_fig / "category_agreement_heatmap.pdf",
    ])

    print(f"\n[done] both figures regenerated from n={n} method-agreement matrices.")


if __name__ == "__main__":
    main()
