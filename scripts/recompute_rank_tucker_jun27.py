#!/usr/bin/env python3
"""
Recompute rank-sensitivity sweep (CP ranks 1-5) and Tucker-vs-CP comparison
on the CLEAN-43 corpus, using the MAIN-pipeline matrix_congruence
(AlignmentTester / CongruenceCoefficient) so that the rank-2 value reproduces
the verified claims-factors phi = 0.062 from outputs/alignment/alignment_results.json.

Writes:
  outputs/expansion/rank_sensitivity.json   (refreshed)
  outputs/expansion/tucker_comparison.json  (refreshed)
and prints a table to stdout.
"""
import sys
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "src"))

from tensor_ops.decomposition import TensorDecomposition
from alignment.congruence import CongruenceCoefficient


def common_symbols():
    claims_meta = json.load(open(BASE / "outputs/nlp/claims_matrix_meta.json"))
    stats_meta = json.load(open(BASE / "outputs/market/stats_matrix_meta.json"))
    factors_meta = json.load(open(BASE / "outputs/tensor/cp_factors_meta.json"))
    cs = claims_meta['symbols']
    ss = stats_meta['symbols']
    fs = factors_meta['symbols']
    common = sorted(set(cs) & set(ss) & set(fs))
    return common, cs, fs


def main():
    claims = np.load(BASE / "outputs/nlp/claims_matrix.npy")
    common, claims_symbols, factors_symbols = common_symbols()
    print(f"Common entities (claims ∩ stats ∩ factors): {len(common)}")

    claims_idx = [claims_symbols.index(s) for s in common]
    factors_idx = [factors_symbols.index(s) for s in common]
    claims_common = claims[claims_idx]

    decomp = TensorDecomposition(
        tensor_dir=BASE / "outputs" / "tensor",
        output_dir=BASE / "outputs" / "tensor",
    )
    cong = CongruenceCoefficient()

    # ---- Rank sweep (CP ranks 1-5) ----
    rank_rows = []
    for rank in range(1, 6):
        factors, exp_var = decomp.cp_decomposition(rank)
        asset_factors = factors[1]              # (49, rank)
        fac_common = asset_factors[factors_idx]  # (43, rank)
        phi = cong.matrix_congruence(claims_common, fac_common)['mean_phi']
        rank_rows.append({
            "rank": rank,
            "explained_variance": float(exp_var),
            "alignment_phi": float(phi),
        })
        print(f"  CP rank {rank}: EV={exp_var*100:.2f}%  phi={phi:.4f}")

    with open(BASE / "outputs/expansion/rank_sensitivity.json", "w") as f:
        json.dump(rank_rows, f, indent=2)

    # ---- Tucker vs CP comparison ----
    cp_rank2 = next(r for r in rank_rows if r["rank"] == 2)

    tucker_ranks = [5, 2, 2]
    core, tfactors, t_exp_var = decomp.tucker_decomposition(tucker_ranks)
    t_asset = tfactors[1]                       # (49, 2)
    t_common = t_asset[factors_idx]
    t_phi = cong.matrix_congruence(claims_common, t_common)['mean_phi']
    print(f"  Tucker {tucker_ranks}: EV={t_exp_var*100:.2f}%  phi={t_phi:.4f}")

    tucker_comp = {
        "cp": {
            "rank": 2,
            "explained_variance": cp_rank2["explained_variance"],
            "alignment_phi": cp_rank2["alignment_phi"],
            "factor_shape": [len(common), 2],
        },
        "tucker": {
            "ranks": tucker_ranks,
            "explained_variance": float(t_exp_var),
            "alignment_phi": float(t_phi),
            "factor_shape": [len(common), 2],
            "core_shape": list(np.array(core).shape),
        },
    }
    with open(BASE / "outputs/expansion/tucker_comparison.json", "w") as f:
        json.dump(tucker_comp, f, indent=2)

    print("\n=== SUMMARY (clean-43, main-pipeline congruence) ===")
    print("Rank sweep:")
    for r in rank_rows:
        print(f"  rank {r['rank']}: EV={r['explained_variance']*100:.2f}%  phi={r['alignment_phi']:.4f}")
    print(f"CP rank-2 phi={cp_rank2['alignment_phi']:.4f}  EV={cp_rank2['explained_variance']*100:.2f}%")
    print(f"Tucker[5,2,2] phi={t_phi:.4f}  EV={t_exp_var*100:.2f}%")


if __name__ == "__main__":
    main()
