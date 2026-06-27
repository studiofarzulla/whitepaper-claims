#!/usr/bin/env python3
"""
TENSOR-DEFI Paper Expansion: Figure Generation

Generates 7 new figures for the expanded paper.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

# Farzulla burgundy
BURGUNDY = '#800020'
GRAY = '#666666'


def fig4_temporal_phi(data_path: Path, output_path: Path):
    """Generate temporal φ evolution plot."""
    print("Generating Figure 4: Temporal φ Evolution...")

    with open(data_path / "temporal_plot_data.json") as f:
        data = json.load(f)

    windows = data['windows']
    mean_phi = data['mean_phi']
    std_phi = data['std_phi']

    # Extract values
    labels = [w['label'] for w in windows]
    phis = [w['phi'] for w in windows]
    window_ids = list(range(len(windows)))

    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot line
    ax.plot(window_ids, phis, 'o-', color=BURGUNDY, linewidth=2, markersize=8)

    # Mean line
    ax.axhline(mean_phi, color=GRAY, linestyle='--', linewidth=1.5, label=f'Mean φ = {mean_phi:.3f}')

    # Confidence band
    ax.fill_between(window_ids, mean_phi - std_phi, mean_phi + std_phi,
                    color=GRAY, alpha=0.2, label=f'±1σ = {std_phi:.3f}')

    ax.set_xlabel('Rolling Window')
    ax.set_ylabel("Tucker's Congruence Coefficient (φ)")
    ax.set_title('Temporal Evolution of Narrative-Market Alignment')
    ax.set_xticks(window_ids)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0.15, 0.30)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path / 'fig4_temporal_phi.pdf', bbox_inches='tight')
    plt.savefig(output_path / 'fig4_temporal_phi.png', bbox_inches='tight', dpi=300)
    plt.close()


def fig5_dimension_alignment(data_path: Path, output_path: Path):
    """Generate per-dimension alignment bar chart."""
    print("Generating Figure 5: Per-Dimension Alignment...")

    with open(data_path / "per_dimension_phi.json") as f:
        data = json.load(f)

    phis = data['claims_stats']['column_phis']

    # Only show non-zero dimensions (first 7 are meaningful)
    valid_idx = [i for i, p in enumerate(phis) if p > 0]
    valid_phis = [phis[i] for i in valid_idx]
    dim_labels = [f'Dim {i+1}' for i in valid_idx]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Color by threshold
    colors = []
    for phi in valid_phis:
        if phi >= 0.85:
            colors.append('#2ecc71')  # Green - similar
        elif phi >= 0.65:
            colors.append('#f39c12')  # Orange - moderate
        else:
            colors.append(BURGUNDY)   # Burgundy - weak

    bars = ax.barh(dim_labels, valid_phis, color=colors, edgecolor='black', linewidth=0.5)

    # Threshold lines
    ax.axvline(0.85, color='#2ecc71', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(0.65, color='#f39c12', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Tucker's Congruence Coefficient (φ)")
    ax.set_ylabel('Aligned Dimension')
    ax.set_title('Per-Dimension Alignment: Claims ↔ Statistics')
    ax.set_xlim(0, 1.0)

    # Legend
    weak_patch = mpatches.Patch(color=BURGUNDY, label='Weak (φ < 0.65)')
    mod_patch = mpatches.Patch(color='#f39c12', label='Moderate (0.65 ≤ φ < 0.85)')
    strong_patch = mpatches.Patch(color='#2ecc71', label='Similar (φ ≥ 0.85)')
    ax.legend(handles=[weak_patch, mod_patch, strong_patch], loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path / 'fig5_dimension_alignment.pdf', bbox_inches='tight')
    plt.savefig(output_path / 'fig5_dimension_alignment.png', bbox_inches='tight', dpi=300)
    plt.close()


def fig6_factor_scatter(data_path: Path, output_path: Path):
    """Generate factor loading scatter plot."""
    print("Generating Figure 6: Factor Loading Scatter...")

    with open(data_path / "factor_loadings.json") as f:
        data = json.load(f)

    with open(data_path / "entity_impact_plot.json") as f:
        entity_data = json.load(f)

    loadings = data['factor_loadings']
    outliers = data['outliers']
    clusters = entity_data['clusters']

    # Create cluster mapping
    cluster_map = {}
    for cluster_id, symbols in clusters.items():
        for sym in symbols:
            cluster_map[sym] = int(cluster_id)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color palette for clusters
    cluster_colors = {0: BURGUNDY, 1: '#3498db', 2: '#2ecc71'}

    # Plot each point
    for sym, vals in loadings.items():
        f1, f2 = vals['Factor_1'], vals['Factor_2']
        cluster = cluster_map.get(sym, 1)  # Default to cluster 1 if not found
        color = cluster_colors.get(cluster, GRAY)

        # Highlight outliers
        if sym in outliers:
            ax.scatter(f1, f2, c=color, s=150, edgecolor='black', linewidth=2, zorder=10)
            ax.annotate(sym, (f1, f2), xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
        else:
            ax.scatter(f1, f2, c=color, s=80, alpha=0.7, edgecolor='white', linewidth=0.5)

    # Axes
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='-')

    ax.set_xlabel('Factor 1 Loading')
    ax.set_ylabel('Factor 2 Loading')
    ax.set_title('Asset Factor Space (CP Decomposition, Rank 2)')

    # Legend
    c0_patch = mpatches.Patch(color=BURGUNDY, label='Cluster 0 (BTC)')
    c1_patch = mpatches.Patch(color='#3498db', label='Cluster 1 (Infrastructure)')
    c2_patch = mpatches.Patch(color='#2ecc71', label='Cluster 2 (DeFi/Smart Contracts)')
    ax.legend(handles=[c0_patch, c1_patch, c2_patch], loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path / 'fig6_factor_scatter.pdf', bbox_inches='tight')
    plt.savefig(output_path / 'fig6_factor_scatter.png', bbox_inches='tight', dpi=300)
    plt.close()


def fig7_rank_sensitivity(data_path: Path, output_path: Path):
    """Generate rank sensitivity dual-axis plot."""
    print("Generating Figure 7: Rank Sensitivity...")

    with open(data_path / "rank_sensitivity.json") as f:
        data = json.load(f)

    ranks = [d['rank'] for d in data]
    variances = [d['explained_variance'] for d in data]
    phis = [d['alignment_phi'] for d in data]

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Variance (left axis)
    color1 = BURGUNDY
    ax1.set_xlabel('CP Decomposition Rank')
    ax1.set_ylabel('Explained Variance', color=color1)
    line1, = ax1.plot(ranks, variances, 'o-', color=color1, linewidth=2, markersize=8, label='Variance')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.7, 1.0)

    # Phi (right axis)
    ax2 = ax1.twinx()
    color2 = '#3498db'
    ax2.set_ylabel('Alignment φ', color=color2)
    line2, = ax2.plot(ranks, phis, 's--', color=color2, linewidth=2, markersize=8, label='φ')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 0.2)

    # Title and legend
    ax1.set_title('Rank Selection: Variance-Alignment Trade-off')
    ax1.set_xticks(ranks)

    # Combined legend - upper left to avoid collision with data points
    lines = [line1, line2]
    labels = ['Explained Variance', 'Alignment φ']
    ax1.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path / 'fig7_rank_sensitivity.pdf', bbox_inches='tight')
    plt.savefig(output_path / 'fig7_rank_sensitivity.png', bbox_inches='tight', dpi=300)
    plt.close()


def fig8_feature_importance(data_path: Path, output_path: Path):
    """Generate feature importance horizontal bar chart."""
    print("Generating Figure 8: Feature Importance...")

    with open(data_path / "feature_importance_plot.json") as f:
        data = json.load(f)

    importance = data['importance']
    features = [d['feature'] for d in importance]
    values = [d['importance'] for d in importance]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Color by sign
    colors = [BURGUNDY if v > 0 else '#3498db' for v in values]

    bars = ax.barh(features, values, color=colors, edgecolor='black', linewidth=0.5)

    # Zero line
    ax.axvline(0, color='black', linewidth=1)

    ax.set_xlabel('Impact on Alignment φ (Ablation)')
    ax.set_ylabel('Claim Category')
    ax.set_title('Feature Importance: Which Claims Drive Alignment?')

    # Invert y-axis so most important is at top
    ax.invert_yaxis()

    # Legend
    pos_patch = mpatches.Patch(color=BURGUNDY, label='Positive (helps alignment)')
    neg_patch = mpatches.Patch(color='#3498db', label='Negative (hurts alignment)')
    ax.legend(handles=[pos_patch, neg_patch], loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path / 'fig8_feature_importance.pdf', bbox_inches='tight')
    plt.savefig(output_path / 'fig8_feature_importance.png', bbox_inches='tight', dpi=300)
    plt.close()


# Dropped during content verification (failed-download stub / wrong-document
# files masquerading as whitepapers); absent from the content-verified corpus.
DROPPED_STUBS = {'YFI', 'AXS', 'SUSHI', 'GALA', 'ENS'}

# Colour by interpretation, not raw sign: the leave-one-out magnitudes are tiny,
# and what matters for the cautionary tale is whether an asset registers as
# helping / hurting alignment at all.
INTERP_COLOUR = {'helps': '#2ecc71', 'hurts': '#e74c3c', 'neutral': GRAY}


def _entity_panel(ax, entities, title, xlim, mark_stubs=False):
    """Draw one leave-one-out entity-impact dot plot on ``ax``."""
    # Sort impact-ascending so highest impact sits at the top of the panel.
    entities = sorted(entities, key=lambda e: e['impact'])
    symbols = [e['symbol'] for e in entities]
    impacts = [e['impact'] for e in entities]
    colours = [INTERP_COLOUR.get(e.get('interpretation', 'neutral'), GRAY)
               for e in entities]

    ypos = list(range(len(symbols)))
    ax.scatter(impacts, ypos, c=colours, s=90, edgecolor='black',
               linewidth=0.8, zorder=10)
    for y, impact, col in zip(ypos, impacts, colours):
        ax.hlines(y, 0, impact, color=col, linewidth=1.4, alpha=0.5)

    # Explicit ticks (robust to draw timing) so stub-marking is deterministic.
    labels = [s + ' *' if (mark_stubs and s in DROPPED_STUBS) else s
              for s in symbols]
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, len(symbols) - 0.3)
    if mark_stubs:
        for tick, sym in zip(ax.get_yticklabels(), symbols):
            if sym in DROPPED_STUBS:
                tick.set_color('#7b241c')
                tick.set_fontweight('bold')

    ax.axvline(0, color='black', linewidth=1.5)
    ax.set_xlim(xlim)
    ax.set_xlabel('Impact on alignment φ (leave-one-out Δφ)')
    ax.set_title(title)


def fig9_entity_impact(data_path: Path, output_path: Path):
    """Two-panel entity-impact figure: contaminated n=37 vs content-verified n=43.

    Left  -- earlier (contaminated) corpus: an apparent XMR/CRV/YFI ``helps''
             ordering, anchored partly by failed-download stub documents.
    Right -- content-verified corpus: no stable specialised-versus-infrastructure
             ordering survives; the apparent split dissolves.
    """
    print("Generating Figure 9: Entity Impact (contaminated vs content-verified)...")

    with open(data_path / "entity_impact_plot.json") as f:
        clean = json.load(f)

    contam_file = data_path / "entity_impact_plot_contaminated_n37.json"
    contaminated = None
    if contam_file.exists():
        with open(contam_file) as f:
            contaminated = json.load(f)

    # Shared symmetric x-axis so magnitudes are directly comparable.
    all_imp = [e['impact'] for e in clean['entities']]
    if contaminated:
        all_imp += [e['impact'] for e in contaminated['entities']]
    span = max(abs(min(all_imp)), abs(max(all_imp)))
    xlim = (-span * 1.18, span * 1.18)

    if contaminated:
        fig, (ax_l, ax_r) = plt.subplots(
            1, 2, figsize=(13, 13), sharex=True)
        n_c = len(contaminated['entities'])
        n_k = len(clean['entities'])
        _entity_panel(
            ax_l, contaminated['entities'],
            f'Earlier corpus (contaminated, n = {n_c})', xlim, mark_stubs=True)
        _entity_panel(
            ax_r, clean['entities'],
            f'Content-verified corpus (n = {n_k})', xlim)
        ax_l.set_ylabel('Asset')
        fig.text(0.5, 0.045,
                 '* failed-download stub / wrong-document files removed in '
                 'content verification (absent from the verified corpus)',
                 ha='center', fontsize=9, color='#7b241c')
        handles = [
            mpatches.Patch(color='#2ecc71', label='Helps alignment'),
            mpatches.Patch(color=GRAY, label='Neutral (|Δφ| negligible)'),
            mpatches.Patch(color='#e74c3c', label='Hurts alignment'),
        ]
        fig.legend(handles=handles, loc='lower center',
                   bbox_to_anchor=(0.5, 0.005), ncol=3, frameon=False)
        fig.suptitle('Entity-Level Alignment Impact: Contamination Reverses the '
                     'Apparent Ordering', y=0.995)
        plt.tight_layout(rect=(0, 0.07, 1, 0.98))
    else:
        # Fallback: single content-verified panel.
        fig, ax = plt.subplots(figsize=(7, 14))
        _entity_panel(ax, clean['entities'],
                      f"Entity-Level Alignment Impact (n = {len(clean['entities'])})",
                      xlim)
        ax.set_ylabel('Asset')
        handles = [
            mpatches.Patch(color='#2ecc71', label='Helps alignment'),
            mpatches.Patch(color=GRAY, label='Neutral (|Δφ| negligible)'),
            mpatches.Patch(color='#e74c3c', label='Hurts alignment'),
        ]
        ax.legend(handles=handles, loc='upper center',
                  bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
        plt.tight_layout()

    plt.savefig(output_path / 'fig9_entity_impact.pdf', bbox_inches='tight')
    plt.savefig(output_path / 'fig9_entity_impact.png', bbox_inches='tight', dpi=300)
    plt.close()


def fig10_tensor_slice(data_path: Path, output_path: Path):
    """Generate tensor slice heatmap."""
    print("Generating Figure 10: Tensor Slice Heatmap...")

    with open(data_path / "tensor_slice.json") as f:
        data = json.load(f)

    slice_data = np.array(data['data'])
    assets = data['assets']
    features = data['features']

    # Clip outliers to [-2, 2] for better visualization (BTC has z-scores ~6.7)
    slice_clipped = np.clip(slice_data, -2, 2)

    # Select subset for readability (every 2nd asset for more coverage)
    asset_idx = list(range(0, len(assets), 2))
    slice_subset = slice_clipped[asset_idx, :]
    asset_labels = [assets[i] for i in asset_idx]

    fig, ax = plt.subplots(figsize=(8, 10))

    # Heatmap with clipped range
    sns.heatmap(slice_subset, annot=False, cmap='RdBu_r', center=0,
                vmin=-2, vmax=2,
                xticklabels=features, yticklabels=asset_labels,
                cbar_kws={'label': 'Z-Score (clipped to ±2)'}, ax=ax)

    ax.set_xlabel('Market Feature')
    ax.set_ylabel('Asset')
    ax.set_title(f'Market Tensor Slice (t = {data["timestamp_index"]}, z-normalized)')

    plt.tight_layout()
    plt.savefig(output_path / 'fig10_tensor_slice.pdf', bbox_inches='tight')
    plt.savefig(output_path / 'fig10_tensor_slice.png', bbox_inches='tight', dpi=300)
    plt.close()


def main():
    """Generate all expansion figures."""
    base_path = Path(__file__).parent.parent
    data_path = base_path / "outputs" / "expansion"
    output_path = base_path / "paper" / "figures"
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TENSOR-DEFI: Expansion Figure Generation")
    print("=" * 60)

    fig4_temporal_phi(data_path, output_path)
    fig5_dimension_alignment(data_path, output_path)
    fig6_factor_scatter(data_path, output_path)
    fig7_rank_sensitivity(data_path, output_path)
    fig8_feature_importance(data_path, output_path)
    fig9_entity_impact(data_path, output_path)
    fig10_tensor_slice(data_path, output_path)

    print("\n" + "=" * 60)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nFigures saved to: {output_path}")
    for f in sorted(output_path.glob("fig*.pdf")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
