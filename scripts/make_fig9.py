#!/usr/bin/env python3
"""
Reproduce Figure 9 (entity-impact cautionary tale) on the clean-43 corpus.

Pipeline:
  1. rebuild_entity_impact_clean43.py  -> outputs/expansion/entity_impact_plot.json
     (clean-43; contaminated n=37 build preserved as *_contaminated_n37.json)
  2. run_expansion_figures.fig9_entity_impact -> two-panel figure
       left  : earlier contaminated corpus (n=37)  -- apparent XMR/CRV/YFI "helps"
       right : content-verified corpus  (n=43)      -- no stable ordering survives

Writes the figure to BOTH the manuscript dir (<repo>/paper/figures) and the
code mirror (<repo>/code/figures).

Usage:  python code/scripts/make_fig9.py
"""

import importlib.util
import shutil
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent          # <repo>/code/scripts
CODE = SCRIPTS.parent                               # <repo>/code
REPO = CODE.parent                                  # <repo>


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    # 1. Refresh the clean-43 JSON the plotter consumes.
    rebuild = _load("rebuild_entity_impact_clean43",
                    SCRIPTS / "rebuild_entity_impact_clean43.py")
    rebuild.main()

    # 2. Plot into the manuscript figure directory.
    figs = _load("run_expansion_figures", SCRIPTS / "run_expansion_figures.py")
    data_path = CODE / "outputs" / "expansion"
    paper_figs = REPO / "paper" / "figures"
    paper_figs.mkdir(parents=True, exist_ok=True)
    figs.fig9_entity_impact(data_path, paper_figs)

    # 3. Mirror into the code/figures copy.
    code_figs = CODE / "figures"
    code_figs.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        shutil.copy2(paper_figs / f"fig9_entity_impact.{ext}",
                     code_figs / f"fig9_entity_impact.{ext}")

    print(f"\nFigure 9 written to:\n  {paper_figs / 'fig9_entity_impact.pdf'}"
          f"\n  {code_figs / 'fig9_entity_impact.pdf'}")


if __name__ == "__main__":
    main()
