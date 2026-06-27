#!/usr/bin/env python3
"""
Inter-model agreement validation: BART-NLI vs DeBERTa-v3 top-1 classification.

PURPOSE
-------
Recomputes the construct-validity reliability number the paper cites as
Cohen's kappa (inter-model top-1 agreement) on the CLEAN, content-verified
n = 43 corpus, using the paper's actual 10-category functional taxonomy.

PROVENANCE / WHY THIS SCRIPT EXISTS
-----------------------------------
The previous `outputs/nlp/model_validation.json` (kappa = 0.142) was produced by
an ad-hoc routine that was never committed, and -- inspecting its stored
`bart_distribution` / `deberta_distribution` -- it ran over an OLD 7--8 label
taxonomy (consensus / technology / security / tokenomics / ...), not the 10
functional categories this paper reports, and over the pre-clean corpus. This
script restores a reproducible path on the correct taxonomy and corpus:

  primary    = facebook/bart-large-mnli         (the paper's primary classifier)
  alternative= cross-encoder/nli-deberta-v3-small (the cited DeBERTa-v3 check)
  labels     = src/nlp/taxonomy.ZERO_SHOT_LABELS (the 10 paper categories)
  corpus     = outputs/nlp/extracted_chunks.json (clean-43)
  sample     = 200 chunks, fixed seed (matches the paper's "random sample of 200
               chunks" design)

Reports exact top-1 agreement, Cohen's kappa on the 10-category top-1 labels,
and relaxed top-3 agreement (alternative top-1 in primary top-3).

Usage:
    python3 scripts/validate_intermodel_kappa.py [--n 200] [--seed 20260627]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent          # .../code
sys.path.insert(0, str(BASE / "src"))
from nlp.taxonomy import ZERO_SHOT_LABELS, LABEL_TO_CATEGORY  # noqa: E402

logging.disable(logging.INFO)

PRIMARY_MODEL = "facebook/bart-large-mnli"
ALT_MODEL = "cross-encoder/nli-deberta-v3-small"


def local_snapshot(repo_id: str) -> str:
    """Resolve the local HF cache snapshot dir for a repo, so the pipeline loads
    it as a *local path* (avoids a transformers tokenizer-patch network call that
    breaks under offline mode). Falls back to the repo id if not cached."""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return repo_id


def load_chunk_pool(chunks_path: Path) -> list[str]:
    data = json.load(open(chunks_path))
    pool = []
    for sym in sorted(data.keys()):
        for ch in data[sym]["chunks"]:
            ch = (ch or "").strip()
            if len(ch) >= 40:          # skip near-empty fragments
                pool.append(ch)
    return pool


def ranked_labels(classifier, texts: list[str]) -> list[list[str]]:
    """Return, for each text, the category keys ranked best-first (top-1 = [0])."""
    out = []
    results = classifier(texts, ZERO_SHOT_LABELS, multi_label=False,
                         truncation=True, batch_size=16)
    if isinstance(results, dict):
        results = [results]
    for r in results:
        ranked = [LABEL_TO_CATEGORY[ZERO_SHOT_LABELS.index(lab)] for lab in r["labels"]]
        out.append(ranked)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260627)
    args = ap.parse_args()

    import torch
    from transformers import pipeline
    from sklearn.metrics import cohen_kappa_score

    device = 0 if torch.cuda.is_available() else -1
    chunks_path = BASE / "outputs" / "nlp" / "extracted_chunks.json"
    pool = load_chunk_pool(chunks_path)

    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(pool))
    idx = rng.choice(len(pool), size=n, replace=False)
    sample = [pool[i] for i in idx]
    print(f"[validate] pool={len(pool)} chunks; sampled n={n} (seed={args.seed})")

    print(f"[validate] loading primary  {PRIMARY_MODEL} ...")
    bart = pipeline("zero-shot-classification", model=local_snapshot(PRIMARY_MODEL), device=device)
    print(f"[validate] loading alternative {ALT_MODEL} ...")
    deberta = pipeline("zero-shot-classification", model=local_snapshot(ALT_MODEL), device=device)

    bart_ranked = ranked_labels(bart, sample)
    deb_ranked = ranked_labels(deberta, sample)

    bart_top1 = [r[0] for r in bart_ranked]
    deb_top1 = [r[0] for r in deb_ranked]

    exact = np.mean([b == d for b, d in zip(bart_top1, deb_top1)])
    kappa = cohen_kappa_score(bart_top1, deb_top1, labels=LABEL_TO_CATEGORY)
    relaxed_top3 = np.mean([d in b3[:3] for d, b3 in zip(deb_top1, bart_ranked)])

    def dist(top1):
        return {c: int(sum(t == c for t in top1)) for c in LABEL_TO_CATEGORY
                if sum(t == c for t in top1) > 0}

    interp = ("almost perfect" if kappa > 0.80 else "substantial" if kappa > 0.60 else
              "moderate" if kappa > 0.40 else "fair" if kappa > 0.20 else
              "slight/poor" if kappa > 0 else "poor")

    payload = {
        "sample_size": int(n),
        "seed": int(args.seed),
        "corpus": "clean-43 (content-verified)",
        "taxonomy": "10 functional categories (src/nlp/taxonomy.py)",
        "primary_model": PRIMARY_MODEL,
        "alternative_model": ALT_MODEL,
        "agreement_rate": float(exact),
        "cohens_kappa": float(kappa),
        "relaxed_top3_agreement": float(relaxed_top3),
        "interpretation": interp,
        "bart_distribution": dist(bart_top1),
        "deberta_distribution": dist(deb_top1),
    }
    out = BASE / "outputs" / "nlp" / "model_validation.json"
    json.dump(payload, open(out, "w"), indent=2)

    print("=" * 60)
    print(f"exact top-1 agreement : {exact:.3f}")
    print(f"Cohen's kappa         : {kappa:.4f}  ({interp})")
    print(f"relaxed top-3         : {relaxed_top3:.3f}")
    print("=" * 60)
    print(f"[validate] wrote {out}")


if __name__ == "__main__":
    main()
