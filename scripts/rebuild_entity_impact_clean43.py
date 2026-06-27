#!/usr/bin/env python3
"""
Rebuild expansion/entity_impact_plot.json from the clean-43 cross-sectional
analysis (whitepaper-claims, FRL reventure).

The figure-generation pipeline (run_expansion_figures.py::fig9_entity_impact)
consumes outputs/expansion/entity_impact_plot.json. That file was a stale n=37
build carrying corpus contamination (YFI/AXS/SUSHI stubs, an apparent
XMR/CRV/YFI "helps" ordering). The content-verified leave-one-out impacts live
in outputs/analysis/cross_sectional_analysis.json (clean-43). This script
projects the clean-43 result into the entity_impact_plot.json schema so the
figure rebuilds on verified data, and preserves the contaminated n=37 build
alongside it (entity_impact_plot_contaminated_n37.json) for the cautionary-tale
two-panel figure.

Schema (matches the n=37 build the plotter expects):
    entities[]  -> {symbol, phi_without, impact, interpretation}  (impact-desc)
    phi_full
    best_aligned, worst_aligned
    clusters
"""

import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "outputs" / "analysis" / "cross_sectional_analysis.json"
DST = BASE / "outputs" / "expansion" / "entity_impact_plot.json"
CONTAM = BASE / "outputs" / "expansion" / "entity_impact_plot_contaminated_n37.json"


def main():
    csa = json.loads(SRC.read_text())

    # Preserve the contaminated n=37 build before overwriting (idempotent).
    if DST.exists() and not CONTAM.exists():
        shutil.copy2(DST, CONTAM)

    entities = csa["entity_analysis"]  # already sorted impact-descending
    out = {
        "entities": entities,
        "phi_full": csa["phi_full"],
        "best_aligned": csa.get("best_aligned", []),
        "worst_aligned": csa.get("worst_aligned", []),
        "clusters": csa.get("clusters", {}),
        "n": len(entities),
        "corpus": "content-verified-43",
        "note": (
            "Rebuilt from outputs/analysis/cross_sectional_analysis.json "
            "(clean-43). Supersedes the n=37 contaminated build retained as "
            "entity_impact_plot_contaminated_n37.json."
        ),
    }
    DST.write_text(json.dumps(out, indent=2) + "\n")

    impacts = [e["impact"] for e in entities]
    symbols = [e["symbol"] for e in entities]
    stubs = {"YFI", "AXS", "SUSHI", "GALA", "ENS"}
    print(f"Wrote {DST}")
    print(f"  n = {len(entities)} | phi_full = {csa['phi_full']:.5f}")
    print(f"  best_aligned = {out['best_aligned']}")
    print(f"  worst_aligned = {out['worst_aligned']}")
    print(f"  top-4: {symbols[:4]}  ({[round(i, 4) for i in impacts[:4]]})")
    print(f"  bottom-4: {symbols[-4:]} ({[round(i, 4) for i in impacts[-4:]]})")
    leaked = sorted(stubs & set(symbols))
    print(f"  dropped-stub assets present (must be []): {leaked}")
    assert not leaked, f"dropped stubs leaked into clean-43: {leaked}"


if __name__ == "__main__":
    main()
