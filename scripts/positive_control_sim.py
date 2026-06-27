#!/usr/bin/env python3
"""
Positive-Control / Power Simulation for the Whitepaper-Claims Procrustes pipeline.

PURPOSE
-------
Answers the sharpest reviewer critique of the null result:

    "A weak-but-significant *same-domain* alignment (stats <-> factors) proves the
     pipeline is *alive*, not that it could *detect* a genuine *cross-domain*
     (claims <-> factors) congruence of magnitude X at n = 43."

This script quantifies the sensitivity of the EXACT pipeline used in the paper.
For a grid of TRUE congruence levels it:

  (1) takes the real clean-43 market FACTOR matrix B (43 x 2 CP asset factors)
      restricted to the 43 content-verified common entities -- the *same* target
      matrix the paper's claims<->factors test uses (or, with --synthetic-factors,
      a matched-covariance Gaussian surrogate of it);

  (2) synthesizes a "claims-like" 43 x 10 matrix A that is a NOISY ORTHOGONAL
      ROTATION of the factor signal embedded in the 10-dim claims space, with
      Gaussian noise calibrated so that the per-factor-column correlation between
      A's signal and B equals the target true_phi (classical signal+noise:
      corr = sqrt(var_signal / (var_signal + var_noise)) = true_phi). Optionally
      the latent signal is further attenuated by the measured claims instrument
      reliability (~0.30) to mirror the real, noisy instrument
      (Spearman attenuation: observed = true * sqrt(rho_claims * rho_factors));

  (3) pushes A and B through the SAME pipeline the paper uses -- the orthogonal
      Procrustes rotation in src/alignment/procrustes.py and the row-permutation
      significance test in src/alignment/congruence.py (CongruenceCoefficient).
      Nothing about the estimator or the test is reinvented here;

  (4) over many Monte-Carlo reps reports the empirical DETECTION RATE
      (power = fraction of reps with permutation p < 0.05), the mean recovered
      pipeline phi (the diluted mean_phi the paper reports), and the mean recovered
      per-factor-column phi (the un-diluted congruence on the 2 real factor axes,
      which is the quantity that maps onto the paper's "True phi").

It then reports the power curve (detection rate vs true_phi at n = 43) and the
minimum detectable effect (MDE): the true_phi at which power first reaches ~80%.

This both quantifies sensitivity (answers critique #2) and provides reproducible
backing for the paper's existing power table (Section "Power Considerations",
14/45/70/90%), which currently has no script.

Usage:
    python3 scripts/positive_control_sim.py                 # default: 500 reps, 200 perms
    python3 scripts/positive_control_sim.py --reps 300 --perms 200
    python3 scripts/positive_control_sim.py --synthetic-factors

Outputs:
    outputs/positive_control.json   (full machine-readable results)
    outputs/positive_control.csv    (tidy power-curve table)
    outputs/positive_control.png    (power-curve figure, if matplotlib available)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --- reuse the EXACT pipeline the paper uses (do NOT reinvent) ----------------
BASE = Path(__file__).resolve().parent.parent          # .../code
sys.path.insert(0, str(BASE / "src"))
from alignment.congruence import CongruenceCoefficient   # noqa: E402

# the pipeline logs an INFO line on every Procrustes call; silence it.
logging.disable(logging.INFO)


# Paper's published power table (Section "Power Considerations", lines ~552-567).
PAPER_POWER_TABLE = {0.30: 0.14, 0.50: 0.45, 0.65: 0.70, 0.70: 0.90}

# Reliability constants used in the paper's disattenuation argument.
RHO_CLAIMS = 0.314    # claims-matrix reliability (mean inter-model correlation,
                      # clean-43 three-way comparison: mean pairwise Pearson r;
                      # see outputs/nlp/method_agreement.json)
RHO_FACTORS = 0.95    # market-data reliability (assumed)


def load_real_factor_matrix():
    """Return (B [43x2], common_symbols) -- the real CP asset factors restricted
    to the 43 content-verified entities common to claims/stats/factors, in the
    same sorted order the pipeline (AlignmentTester.load_matrices) uses."""
    out = BASE / "outputs"
    fac = np.load(out / "tensor" / "cp_asset_factors.npy")
    fsym = json.load(open(out / "tensor" / "cp_factors_meta.json"))["symbols"]
    csym = json.load(open(out / "nlp" / "claims_matrix_meta.json"))["symbols"]
    ssym = json.load(open(out / "market" / "stats_matrix_meta.json"))["symbols"]
    common = sorted(set(fsym) & set(csym) & set(ssym))
    B = fac[[fsym.index(s) for s in common]].astype(float)
    return B, common


def matched_covariance_surrogate(B, rng):
    """Gaussian surrogate with the same per-column mean/covariance as the real
    factor matrix B (used when --synthetic-factors is passed)."""
    mu = B.mean(axis=0)
    cov = np.cov(B, rowvar=False)
    return rng.multivariate_normal(mu, cov, size=B.shape[0])


def synth_claims_like(B, true_phi, rng, embed_dim=10, reliability=1.0):
    """Synthesize a claims-like (n x embed_dim) matrix that is a noisy orthogonal
    rotation of the factor matrix B (n x k), calibrated to the target congruence.

    Construction
    ------------
    1. For each factor column k: center it (fc_k) and add iid Gaussian noise with
       variance var(fc_k) * (1 - phi^2) / phi^2, so that
            corr(fc_k + noise, fc_k) = phi   (in population).
       phi is the *effective* embedded correlation -- the target true_phi times
       the measurement attenuation sqrt(reliability * RHO_FACTORS) when a noisy
       instrument is being mimicked (reliability < 1).
    2. Stack the corrupted columns -> H (n x k) and embed them into the
       embed_dim-dimensional claims space via a random orthonormal map U
       (U^T U = I_k): A = H @ U^T. This is an exact (isometric) rotation of the
       k-dim factor signal into a k-dim subspace of the claims space -- i.e. a
       "noisy orthogonal rotation of the factors".

    Returns the (n x embed_dim) claims-like matrix.
    """
    n, k = B.shape
    phi = true_phi * np.sqrt(reliability * RHO_FACTORS) if reliability < 1.0 else true_phi
    phi = float(np.clip(phi, 1e-6, 0.999999))

    H = np.empty((n, k))
    for j in range(k):
        fc = B[:, j] - B[:, j].mean()
        var_fc = float(np.var(fc))
        if var_fc <= 0:
            H[:, j] = rng.standard_normal(n)
            continue
        noise_var = var_fc * (1.0 - phi ** 2) / (phi ** 2)
        H[:, j] = fc + rng.standard_normal(n) * np.sqrt(noise_var)

    # random orthonormal embedding U (embed_dim x k), columns orthonormal
    G = rng.standard_normal((embed_dim, k))
    U, _ = np.linalg.qr(G)          # U: embed_dim x k, U^T U = I_k
    A = H @ U.T                      # n x embed_dim
    return A


def run_cell(B, true_phi, reps, perms, reliability, cc, rng):
    """Run the Monte-Carlo power estimate for one (true_phi, reliability) cell.

    Returns dict with power, mean recovered pipeline phi, mean recovered
    per-factor-column phi, and Monte-Carlo standard errors.
    """
    n_real_cols = B.shape[1]
    detections = np.empty(reps, dtype=bool)
    obs_phi = np.empty(reps)        # diluted pipeline mean_phi (what the paper reports)
    real_phi = np.empty(reps)       # un-diluted congruence on the real factor columns

    for r in range(reps):
        A = synth_claims_like(B, true_phi, rng, embed_dim=10, reliability=reliability)
        # same permutation significance test the pipeline uses
        pt = cc.permutation_test(A, B, n_permutations=perms)
        detections[r] = pt["p_value"] < 0.05
        obs_phi[r] = pt["observed_phi"]
        # recovered congruence on the genuine factor axes (un-diluted by zero-padding)
        cols = cc.matrix_congruence(A, B)["column_phis"]
        real_phi[r] = float(np.mean(np.abs(cols[:n_real_cols])))

    power = float(detections.mean())
    return {
        "true_phi": float(true_phi),
        "reliability": float(reliability),
        "reps": int(reps),
        "perms": int(perms),
        "power": power,
        "power_se": float(np.sqrt(power * (1 - power) / reps)),
        "mean_recovered_phi_pipeline": float(obs_phi.mean()),
        "mean_recovered_phi_factor_cols": float(real_phi.mean()),
        "n_detections": int(detections.sum()),
    }


def interp_mde(phis, powers, target=0.80):
    """Linear-interpolate the true_phi at which power first crosses `target`.
    Returns None if the curve never reaches target within the grid."""
    phis = np.asarray(phis, float)
    powers = np.asarray(powers, float)
    order = np.argsort(phis)
    phis, powers = phis[order], powers[order]
    if powers.max() < target:
        return None
    for i in range(1, len(phis)):
        if powers[i - 1] < target <= powers[i]:
            p0, p1 = powers[i - 1], powers[i]
            x0, x1 = phis[i - 1], phis[i]
            if p1 == p0:
                return float(x1)
            return float(x0 + (target - p0) * (x1 - x0) / (p1 - p0))
    # already above target at the smallest grid point
    if powers[0] >= target:
        return float(phis[0])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=500,
                    help="Monte-Carlo reps per cell (default 500, matches paper)")
    ap.add_argument("--perms", type=int, default=200,
                    help="permutations per significance test (default 200, matches paper)")
    ap.add_argument("--grid", type=float, nargs="+",
                    default=[0.2, 0.3, 0.5, 0.65, 0.8],
                    help="true_phi grid (default 0.2 0.3 0.5 0.65 0.8)")
    ap.add_argument("--seed", type=int, default=20260627)
    ap.add_argument("--synthetic-factors", action="store_true",
                    help="use a matched-covariance Gaussian surrogate instead of the real factors")
    ap.add_argument("--realistic-reliability", type=float, default=RHO_CLAIMS,
                    help="claims reliability for the realistic-instrument scenario (default 0.30)")
    args = ap.parse_args()

    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    cc = CongruenceCoefficient()

    B_real, common = load_real_factor_matrix()
    if args.synthetic_factors:
        B = matched_covariance_surrogate(B_real, rng)
        factor_source = "matched-covariance Gaussian surrogate"
    else:
        B = B_real
        factor_source = "real CP asset factors (clean-43)"

    n, k = B.shape
    print(f"[positive-control] n={n} entities, factor matrix {B.shape} ({factor_source})")
    print(f"[positive-control] reps={args.reps}, perms={args.perms}, grid={args.grid}")
    print(f"[positive-control] factor-column correlation corr(f0,f1) = "
          f"{np.corrcoef(B[:,0], B[:,1])[0,1]:.3f}")

    scenarios = {
        # headline: true_phi is exactly the congruence the pipeline sees
        "ideal_reliability_1.0": 1.0,
        # realistic: latent congruence attenuated by the measured claims instrument
        f"realistic_reliability_{args.realistic_reliability:g}": args.realistic_reliability,
    }

    results = {}
    for sname, rel in scenarios.items():
        print(f"\n[positive-control] === scenario: {sname} (reliability={rel}) ===")
        cells = []
        for phi in args.grid:
            cell = run_cell(B, phi, args.reps, args.perms, rel, cc, rng)
            cells.append(cell)
            print(f"  true_phi={phi:>4}  power={cell['power']:.3f} "
                  f"(+-{cell['power_se']:.3f})  "
                  f"recovered_phi[factor-cols]={cell['mean_recovered_phi_factor_cols']:.3f}  "
                  f"recovered_phi[pipeline]={cell['mean_recovered_phi_pipeline']:.3f}")
        mde = interp_mde([c["true_phi"] for c in cells],
                         [c["power"] for c in cells], target=0.80)
        results[sname] = {"reliability": rel, "cells": cells, "mde_power80": mde}
        print(f"  --> MDE (80% power) = "
              f"{('%.3f' % mde) if mde is not None else 'NOT REACHED on grid'}")

    # ---- comparison to the paper's published power table -------------------
    ideal_cells = results["ideal_reliability_1.0"]["cells"]
    ideal_by_phi = {round(c["true_phi"], 3): c["power"] for c in ideal_cells}
    paper_cmp = []
    for phi, paper_pow in sorted(PAPER_POWER_TABLE.items()):
        sim_pow = ideal_by_phi.get(round(phi, 3))
        paper_cmp.append({
            "true_phi": phi,
            "paper_power": paper_pow,
            "sim_power": sim_pow,
            "abs_diff": (abs(sim_pow - paper_pow) if sim_pow is not None else None),
        })

    runtime = time.time() - t_start
    payload = {
        "description": "Positive-control / power simulation for the whitepaper-claims "
                       "Procrustes + permutation pipeline. Quantifies detection power vs "
                       "true cross-domain congruence at n=43 and the minimum detectable effect.",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(runtime, 1),
        "config": {
            "reps": args.reps,
            "perms": args.perms,
            "grid": args.grid,
            "seed": args.seed,
            "factor_source": factor_source,
            "n_entities": n,
            "n_factor_columns": k,
            "claims_dim": 10,
            "rho_claims": RHO_CLAIMS,
            "rho_factors": RHO_FACTORS,
            "alpha": 0.05,
            "common_symbols": common,
        },
        "pipeline": {
            "estimator": "src/alignment/procrustes.py :: ProcrustesAlignment.orthogonal_procrustes",
            "test": "src/alignment/congruence.py :: CongruenceCoefficient.permutation_test "
                    "(row-permutation of target, one-tailed p on mean Tucker phi)",
            "note": "pipeline reused as-is; nothing re-implemented",
        },
        "scenarios": results,
        "paper_power_table": PAPER_POWER_TABLE,
        "paper_comparison_ideal_scenario": paper_cmp,
        "headline": {
            "mde_power80_ideal": results["ideal_reliability_1.0"]["mde_power80"],
            "mde_power80_realistic": results[
                f"realistic_reliability_{args.realistic_reliability:g}"]["mde_power80"],
        },
    }

    out = BASE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "positive_control.json"
    json.dump(payload, open(json_path, "w"), indent=2)
    print(f"\n[positive-control] wrote {json_path}")

    # tidy CSV
    csv_path = out / "positive_control.csv"
    with open(csv_path, "w") as f:
        f.write("scenario,reliability,true_phi,power,power_se,"
                "recovered_phi_factor_cols,recovered_phi_pipeline,reps,perms\n")
        for sname, sres in results.items():
            for c in sres["cells"]:
                f.write(f"{sname},{c['reliability']},{c['true_phi']},{c['power']:.4f},"
                        f"{c['power_se']:.4f},{c['mean_recovered_phi_factor_cols']:.4f},"
                        f"{c['mean_recovered_phi_pipeline']:.4f},{c['reps']},{c['perms']}\n")
    print(f"[positive-control] wrote {csv_path}")

    # power-curve figure (best-effort)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for sname, sres in results.items():
            cells = sorted(sres["cells"], key=lambda c: c["true_phi"])
            xs = [c["true_phi"] for c in cells]
            ys = [c["power"] for c in cells]
            es = [c["power_se"] for c in cells]
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=sname)
        # paper points
        pp = sorted(PAPER_POWER_TABLE.items())
        ax.scatter([p for p, _ in pp], [q for _, q in pp], marker="x", s=70,
                   color="black", zorder=5, label="paper table")
        ax.axhline(0.80, ls="--", color="grey", lw=1)
        ax.text(args.grid[0], 0.81, "80% power", fontsize=8, color="grey")
        ax.set_xlabel(r"true congruence $\phi$")
        ax.set_ylabel(r"detection rate (power) at $n=43$, $\alpha=0.05$")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Positive-control power curve: Procrustes + permutation pipeline")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        png_path = out / "positive_control.png"
        fig.savefig(png_path, dpi=150)
        print(f"[positive-control] wrote {png_path}")
    except Exception as e:  # pragma: no cover
        print(f"[positive-control] (figure skipped: {e})")

    # ---- console summary ----------------------------------------------------
    print("\n" + "=" * 72)
    print("POWER CURVE (ideal scenario, true_phi = pipeline-visible congruence)")
    print("=" * 72)
    print(f"{'true_phi':>9} | {'sim power':>10} | {'paper power':>11} | recovered phi (factor cols)")
    for c in ideal_cells:
        pp = PAPER_POWER_TABLE.get(round(c["true_phi"], 3))
        pp_s = f"{pp:.0%}" if pp is not None else "   --"
        print(f"{c['true_phi']:>9} | {c['power']:>9.1%} | {pp_s:>11} | "
              f"{c['mean_recovered_phi_factor_cols']:.3f}")
    mde_i = results["ideal_reliability_1.0"]["mde_power80"]
    mde_r = results[f"realistic_reliability_{args.realistic_reliability:g}"]["mde_power80"]
    print("-" * 72)
    print(f"MDE (80% power), ideal instrument     : "
          f"{('phi = %.3f' % mde_i) if mde_i else 'not reached on grid'}")
    print(f"MDE (80% power), realistic instrument : "
          f"{('phi = %.3f' % mde_r) if mde_r else 'not reached on grid'}  "
          f"(claims reliability = {args.realistic_reliability})")
    print(f"runtime: {runtime:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
