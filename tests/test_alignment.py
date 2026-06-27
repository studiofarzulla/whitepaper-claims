"""Tests for alignment metrics (alternative_metrics.py, congruence.py)."""
import numpy as np
import pytest

from alignment.alternative_metrics import (
    rv_coefficient,
    distance_correlation,
    bootstrap_metric,
)
from alignment.congruence import CongruenceCoefficient


# ── rv_coefficient ──────────────────────────────────────────────────

class TestRVCoefficient:
    def test_identical_matrices(self, rng):
        """RV(X, X) = 1."""
        X = rng.randn(30, 5)
        assert rv_coefficient(X, X) == pytest.approx(1.0, abs=1e-10)

    def test_range_01(self, rng):
        """RV ∈ [0, 1] for arbitrary matrices."""
        X = rng.randn(30, 5)
        Y = rng.randn(30, 5)
        rv = rv_coefficient(X, Y)
        assert 0.0 <= rv <= 1.0 + 1e-10

    def test_symmetry(self, rng):
        """RV(X, Y) = RV(Y, X)."""
        X = rng.randn(30, 5)
        Y = rng.randn(30, 5)
        assert rv_coefficient(X, Y) == pytest.approx(rv_coefficient(Y, X), abs=1e-10)

    def test_uncorrelated_low(self):
        """Independent random matrices → RV well below 1."""
        np.random.seed(123)
        X = np.random.randn(100, 5)
        Y = np.random.randn(100, 5)
        rv = rv_coefficient(X, Y)
        assert rv < 0.3

    def test_zero_matrix(self, rng):
        """Zero matrix → RV = 0."""
        X = rng.randn(20, 4)
        Y = np.zeros((20, 4))
        assert rv_coefficient(X, Y) == 0.0


# ── distance_correlation ───────────────────────────────────────────

class TestDistanceCorrelation:
    def test_identical(self, rng):
        """dCor(X, X) = 1."""
        X = rng.randn(30, 3)
        assert distance_correlation(X, X) == pytest.approx(1.0, abs=0.01)

    def test_symmetry(self, rng):
        """dCor(X, Y) = dCor(Y, X)."""
        X = rng.randn(30, 3)
        Y = rng.randn(30, 4)
        assert distance_correlation(X, Y) == pytest.approx(
            distance_correlation(Y, X), abs=1e-10
        )

    def test_range_01(self, rng):
        """dCor ∈ [0, 1]."""
        X = rng.randn(30, 3)
        Y = rng.randn(30, 3)
        dc = distance_correlation(X, Y)
        assert 0.0 <= dc <= 1.0 + 1e-10

    def test_independent_near_zero(self):
        """Large independent samples → dCor near 0."""
        np.random.seed(99)
        X = np.random.randn(200, 2)
        Y = np.random.randn(200, 2)
        dc = distance_correlation(X, Y)
        assert dc < 0.2

    def test_linearly_related(self):
        """Linearly related data → dCor near 1."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        Y = X * 3.0 + 0.01 * np.random.randn(100, 2)
        dc = distance_correlation(X, Y)
        assert dc > 0.9


# ── CongruenceCoefficient.tuckers_phi ──────────────────────────────

class TestTuckersPhi:
    def setup_method(self):
        self.cc = CongruenceCoefficient()

    def test_identical_vectors(self):
        """φ(x, x) = 1."""
        x = np.array([1.0, 2.0, 3.0])
        assert self.cc.tuckers_phi(x, x) == pytest.approx(1.0, abs=1e-12)

    def test_opposite_vectors(self):
        """φ(x, -x) = -1."""
        x = np.array([1.0, 2.0, 3.0])
        assert self.cc.tuckers_phi(x, -x) == pytest.approx(-1.0, abs=1e-12)

    def test_orthogonal_vectors(self):
        """φ of orthogonal vectors = 0."""
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        assert self.cc.tuckers_phi(x, y) == pytest.approx(0.0, abs=1e-12)

    def test_zero_vector(self):
        """φ with zero vector = 0."""
        x = np.array([1.0, 2.0])
        y = np.zeros(2)
        assert self.cc.tuckers_phi(x, y) == 0.0

    def test_range_minus1_to_1(self, rng):
        """φ ∈ [-1, 1] for arbitrary vectors."""
        for _ in range(10):
            x = rng.randn(20)
            y = rng.randn(20)
            phi = self.cc.tuckers_phi(x, y)
            assert -1.0 - 1e-10 <= phi <= 1.0 + 1e-10


# ── CongruenceCoefficient.matrix_congruence ────────────────────────

class TestMatrixCongruence:
    def setup_method(self):
        self.cc = CongruenceCoefficient()

    def test_identity_self(self):
        """Congruence of a matrix with itself → high φ."""
        np.random.seed(42)
        A = np.random.randn(30, 5)
        result = self.cc.matrix_congruence(A, A)
        # After Procrustes self-alignment, should be near 1
        assert result['mean_phi'] > 0.95

    def test_returns_expected_keys(self, rng):
        A = rng.randn(20, 4)
        B = rng.randn(20, 4)
        result = self.cc.matrix_congruence(A, B)
        assert set(result.keys()) >= {'mean_phi', 'rms_phi', 'column_phis', 'interpretation', 'alignment'}

    def test_per_column_phis_count(self, rng):
        """Number of column phis matches number of columns."""
        A = rng.randn(25, 6)
        B = rng.randn(25, 6)
        result = self.cc.matrix_congruence(A, B)
        assert len(result['column_phis']) == 6

    def test_interpretation_values(self, rng):
        """Interpretation string is one of the valid categories."""
        A = rng.randn(20, 4)
        B = rng.randn(20, 4)
        result = self.cc.matrix_congruence(A, B)
        assert result['interpretation'] in {'equivalent', 'similar', 'moderate', 'weak'}


# ── bootstrap_metric ───────────────────────────────────────────────

class TestBootstrapMetric:
    def test_returns_expected_keys(self, rng):
        X = rng.randn(40, 3)
        Y = rng.randn(40, 3)
        result = bootstrap_metric(X, Y, rv_coefficient, n_bootstrap=200)
        assert set(result.keys()) >= {'ci_lower', 'ci_upper', 'se'}

    def test_ci_ordering(self, rng):
        """ci_lower < ci_upper."""
        X = rng.randn(40, 3)
        Y = rng.randn(40, 3)
        result = bootstrap_metric(X, Y, rv_coefficient, n_bootstrap=200)
        assert result['ci_lower'] < result['ci_upper']

    def test_se_nonnegative(self, rng):
        X = rng.randn(40, 3)
        Y = rng.randn(40, 3)
        result = bootstrap_metric(X, Y, rv_coefficient, n_bootstrap=200)
        assert result['se'] >= 0
