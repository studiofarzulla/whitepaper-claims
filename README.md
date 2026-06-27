# whitepaper-claims: Are Whitepaper Claims Reflected in Market Structure?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17917922.svg)](https://doi.org/10.5281/zenodo.17917922)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Are Whitepaper Claims Reflected in Market Structure? A Contamination-Aware Pipeline and a Power-Limited Null**

*Murad Farzulla • [Dissensus](https://dissensus.ai) • 2026*

---

## Overview

A reproducible, **content-verified** pipeline for measuring how closely a cryptocurrency project's stated whitepaper narrative corresponds to how its token actually behaves in markets — together with a **positive-control / power apparatus** that states what the design can and cannot detect.

The pipeline couples:

- **Zero-shot NLP classification** (BART-MNLI, cross-validated against a sentence-embedding classifier and an instruction-tuned LLM) of whitepaper text into 10 functional categories;
- a panel of **cross-sectional market-structure statistics** computed from two years of hourly data;
- **Procrustes alignment** with Tucker's congruence coefficient (φ); and
- a **positive-control simulation** calibrated on the measured reliability of the text instrument, which fixes the minimum detectable effect.

This is framed as a **method plus a cautionary tale**, not a verdict about narratives and prices.

### Two results

**1. A contamination cautionary tale.** An earlier version of the corpus produced a clean, plausible cross-sectional finding — specialised single-purpose tokens appearing to align with their stated narrative while broad infrastructure tokens did not. That finding was, in full, an artifact of corpus contamination: roughly a quarter of the documents were failed-download stubs or wrong-document files (in one case the "Cosmos" whitepaper was actually Binance Smart Chain text). Content-level verification (word count, project-name density, right-document confirmation) dissolves it — on the clean 43-asset corpus **no token registers as helping alignment** and no specialised-versus-infrastructure structure survives.

**2. A power-limited null.** With a low-reliability text instrument and n = 43, the realistic minimum detectable effect is large. The design can reject *strong* narrative–market alignment but cannot adjudicate weak-to-moderate alignment — **absence of evidence, not evidence of absence.**

| Result | Value |
|--------|-------|
| Claims–statistics congruence φ | **0.303** dimension-matched / **0.223** zero-padded — both non-significant (p ≈ 0.35–0.46) |
| Realistic detection floor (MDE, 80% power) | **φ ≈ 0.66** |
| Inter-method reliability (Cohen's κ) | **0.25** (fair) |
| Content-verified corpus | **43** whitepapers ∩ hourly market coverage |
| Market observations | **17,543** hourly timestamps (2023–2024) |

> **Note on the factor leg.** An earlier draft reported a claims↔market-*factor* alignment from a CP tensor decomposition. That leg is **not reported**: global per-feature normalisation makes Bitcoin a multi-σ outlier dominating the leading factor, so the factor space is degenerate for raw-scale features. The CP/tensor machinery is retained here only as the calibration base for the positive control (see the paper's methodology appendix); the headline claims–statistics result does not use it.

## Repository Structure

```
whitepaper-claims/
├── src/                      # Python modules
│   ├── nlp/                  # zero-shot / embedding / LLM classification
│   ├── alignment/            # Procrustes alignment, Tucker's φ
│   ├── tensor_ops/           # CP decomposition (positive-control calibration base)
│   ├── market/               # market-statistics construction
│   ├── stats/                # statistical tests
│   └── visualization/        # plotting
├── scripts/                  # analysis pipeline + verification/regeneration scripts
├── data/
│   ├── whitepapers/          # content-verified PDF corpus
│   └── market/               # hourly market data (Parquet)
├── outputs/                  # pipeline outputs (NLP, alignment, analysis, positive_control)
├── figures/                  # generated figures
├── tests/                    # pytest suite
├── CITATION.cff
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/studiofarzulla/whitepaper-claims.git
cd whitepaper-claims
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Hardware

- **RAM:** 16GB minimum (32GB recommended)
- **GPU:** optional but recommended for NLP inference (CUDA / ROCm supported)
- **Storage:** ~200MB with full data

## Reproducibility

All inputs are content-verified and uniformly on the clean 43-asset corpus. Key entry points:

```bash
# Full pipeline (NLP → statistics → alignment → figures)
python scripts/run_full_pipeline.py

# Positive control / power analysis (sets the MDE)
python scripts/positive_control_sim.py

# Inter-method reliability (Cohen's κ on the 10-category taxonomy)
python scripts/validate_intermodel_kappa.py

# Entity-impact figure (clean-43 vs the contaminated n=37 panel)
python scripts/make_fig9.py
```

The corpus-contamination diagnosis is auditable: `outputs/expansion/entity_impact_plot_contaminated_n37.json` preserves the earlier contaminated build (clearly labelled) for the before/after comparison, while every reported number derives from the clean-43 outputs.

## Citation

```bibtex
@misc{farzulla2026whitepaper,
  title  = {Are Whitepaper Claims Reflected in Market Structure? A Contamination-Aware Pipeline and a Power-Limited Null},
  author = {Farzulla, Murad},
  year   = {2026},
  doi    = {10.5281/zenodo.17917922}
}
```

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

- **Author:** Murad Farzulla
- **ORCID:** [0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)
- **Lab:** [Dissensus](https://dissensus.ai)
- **GitHub:** [@studiofarzulla](https://github.com/studiofarzulla)

---

*Part of [Dissensus](https://dissensus.ai) — research on adversarial systems and complexity across finance, governance, and AI.*
