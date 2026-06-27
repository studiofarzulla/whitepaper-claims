"""Tests for Random Matrix Theory analysis (src/stats/rmt.py)."""
import numpy as np
import pytest
from scipy import integrate

from stats.rmt import (
    marchenko_pastur_bounds,
    marchenko_pastur_pdf,
    tracy_widom_test,
    eigenvalue_ratio_test,
    full_rmt_analysis,
    test_against_mp as run_mp_test,
)


# ── marchenko_pastur_bounds ─────────────────────────────────────────

class TestMarchenkoPasturBounds:
    def test_gamma_one(self):
        """γ=1 (square) ⇒ bounds (0, 4)."""
        lb, ub = marchenko_pastur_bounds(100, 100)
        assert lb == pytest.approx(0.0, abs=1e-12)
        assert ub == pytest.approx(4.0, abs=1e-12)

    def test_small_ratio(self):
        """n << p ⇒ narrow bulk around σ²."""
        lb, ub = marchenko_pastur_bounds(10, 10000)
        assert 0 <= lb < ub

    def test_large_ratio(self):
        """n >> p ⇒ uses p/n as aspect ratio."""
        lb, ub = marchenko_pastur_bounds(10000, 10)
        assert 0 <= lb < ub

    def test_sigma_scaling(self):
        """Bounds scale with σ²."""
        lb1, ub1 = marchenko_pastur_bounds(100, 100, sigma=1.0)
        lb2, ub2 = marchenko_pastur_bounds(100, 100, sigma=2.0)
        assert ub2 == pytest.approx(4 * ub1, rel=1e-10)
        assert lb2 == pytest.approx(4 * lb1, abs=1e-12)

    def test_lower_bound_nonnegative(self):
        """Lower bound is always ≥ 0."""
        for n, p in [(5, 100), (100, 5), (50, 50)]:
            lb, _ = marchenko_pastur_bounds(n, p)
            assert lb >= 0

    def test_upper_gt_lower(self):
        """Upper bound always exceeds lower bound."""
        for n, p in [(10, 50), (50, 10), (30, 30)]:
            lb, ub = marchenko_pastur_bounds(n, p)
            assert ub > lb


# ── marchenko_pastur_pdf ────────────────────────────────────────────

class TestMarchenkoPasturPDF:
    def test_normalization(self):
        """PDF integrates to approximately 1 over support."""
        n, p = 100, 200
        lb, ub = marchenko_pastur_bounds(n, p)
        x = np.linspace(lb + 1e-8, ub - 1e-8, 5000)
        pdf = marchenko_pastur_pdf(x, n, p)
        area = np.trapezoid(pdf, x)
        assert area == pytest.approx(1.0, abs=0.05)

    def test_zero_outside_bounds(self):
        """PDF is zero outside MP support."""
        n, p = 50, 100
        _, ub = marchenko_pastur_bounds(n, p)
        x_outside = np.array([ub + 1.0, ub + 10.0, -0.5])
        pdf = marchenko_pastur_pdf(x_outside, n, p)
        np.testing.assert_array_equal(pdf, 0.0)

    def test_nonnegative(self):
        """PDF is non-negative everywhere."""
        n, p = 60, 120
        lb, ub = marchenko_pastur_bounds(n, p)
        x = np.linspace(lb, ub, 500)
        pdf = marchenko_pastur_pdf(x, n, p)
        assert np.all(pdf >= 0)

    def test_shape_matches_input(self):
        """Output array has same shape as input."""
        x = np.linspace(0, 5, 200)
        pdf = marchenko_pastur_pdf(x, 50, 100)
        assert pdf.shape == x.shape

    def test_sigma_scaling_pdf(self):
        """Changing σ shifts/scales the density."""
        n, p = 80, 160
        lb1, ub1 = marchenko_pastur_bounds(n, p, sigma=1.0)
        lb2, ub2 = marchenko_pastur_bounds(n, p, sigma=2.0)
        mid1 = (lb1 + ub1) / 2
        mid2 = (lb2 + ub2) / 2
        # Midpoint of sigma=2 support is 4x that of sigma=1
        assert mid2 == pytest.approx(4 * mid1, rel=0.01)


# ── tracy_widom_test ────────────────────────────────────────────────

class TestTracyWidomTest:
    def test_inside_bounds_not_significant(self):
        """Eigenvalue well inside MP bounds → not significant."""
        _, ub = marchenko_pastur_bounds(100, 200)
        result = tracy_widom_test(ub * 0.5, 100, 200)
        assert result['significant'] == False

    def test_far_above_bounds_significant(self):
        """Eigenvalue far above MP bounds → significant."""
        _, ub = marchenko_pastur_bounds(100, 200)
        result = tracy_widom_test(ub * 10, 100, 200)
        assert result['significant'] == True

    def test_returns_expected_keys(self):
        """Result dict contains all expected keys."""
        result = tracy_widom_test(5.0, 50, 100)
        assert set(result.keys()) >= {'z_score', 'p_value', 'significant', 'alpha', 'mu', 'scale'}

    def test_p_value_range(self):
        """p-value is in [0, 1]."""
        result = tracy_widom_test(3.0, 50, 100)
        assert 0 <= result['p_value'] <= 1

    def test_custom_alpha(self):
        """Significance respects custom α."""
        result = tracy_widom_test(5.0, 50, 100, alpha=0.5)
        assert result['alpha'] == 0.5


# ── eigenvalue_ratio_test ───────────────────────────────────────────

class TestEigenvalueRatioTest:
    def test_identity_eigenvalues(self):
        """Identity matrix eigenvalues (all 1s) → ratios ≈ 1."""
        eigvals = np.ones(10)
        result = eigenvalue_ratio_test(eigvals)
        assert result['max_ratio'] == pytest.approx(1.0, abs=0.01)

    def test_clear_factor_structure(self):
        """Large gap between signal and noise eigenvalues."""
        eigvals = np.array([50.0, 45.0, 2.0, 1.5, 1.0, 0.8])
        result = eigenvalue_ratio_test(eigvals)
        # Gap should be between position 1 and 2 (45 / 2 = 22.5)
        assert result['suggested_n_factors'] == 2
        assert result['max_ratio'] > 10

    def test_returns_expected_keys(self):
        result = eigenvalue_ratio_test(np.array([5.0, 3.0, 1.0]))
        assert set(result.keys()) >= {'ratios', 'max_ratio', 'max_ratio_position', 'suggested_n_factors'}

    def test_descending_order_enforced(self):
        """Function sorts eigenvalues regardless of input order."""
        eigvals_asc = np.array([1.0, 2.0, 5.0, 10.0])
        eigvals_desc = np.array([10.0, 5.0, 2.0, 1.0])
        r1 = eigenvalue_ratio_test(eigvals_asc)
        r2 = eigenvalue_ratio_test(eigvals_desc)
        assert r1['max_ratio'] == pytest.approx(r2['max_ratio'], rel=1e-10)


# ── test_against_mp ─────────────────────────────────────────────────

class TestTestAgainstMP:
    def test_random_matrix_few_signals(self, rng):
        """Pure random matrix should have few/no signal eigenvalues."""
        X = rng.randn(100, 50)
        result = run_mp_test(X)
        # Most eigenvalues should fall within MP bounds for pure noise
        assert result['n_signal'] <= 5

    def test_matrix_with_planted_signal(self, rng):
        """Matrix with planted signal should have signal eigenvalues."""
        X = rng.randn(100, 50)
        # Plant a strong rank-1 signal
        u = rng.randn(100, 1)
        v = rng.randn(1, 50)
        X += 10 * u @ v
        result = run_mp_test(X)
        assert result['n_signal'] >= 1

    def test_returns_expected_keys(self, rng):
        X = rng.randn(30, 20)
        result = run_mp_test(X)
        expected = {'eigenvalues', 'lambda_plus', 'lambda_minus', 'n_signal',
                    'n_below', 'n_total', 'signal_variance_ratio',
                    'noise_variance_ratio', 'matrix_shape'}
        assert set(result.keys()) >= expected


# ── full_rmt_analysis ───────────────────────────────────────────────

class TestFullRMTAnalysis:
    def test_returns_expected_keys(self, rng):
        X = rng.randn(40, 20)
        result = full_rmt_analysis(X, name="test")
        assert set(result.keys()) >= {'name', 'mp', 'tw', 'ratio', 'n_signal_consensus'}

    def test_name_propagated(self, rng):
        X = rng.randn(40, 20)
        result = full_rmt_analysis(X, name="my_matrix")
        assert result['name'] == "my_matrix"

    def test_sub_results_are_dicts(self, rng):
        X = rng.randn(40, 20)
        result = full_rmt_analysis(X)
        assert isinstance(result['mp'], dict)
        assert isinstance(result['tw'], dict)
        assert isinstance(result['ratio'], dict)
