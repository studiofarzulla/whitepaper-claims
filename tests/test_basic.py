"""Import smoke-tests for all src modules used in the test suite."""
import pytest


def test_import_stats_rmt():
    from stats.rmt import (
        marchenko_pastur_bounds,
        marchenko_pastur_pdf,
        tracy_widom_test,
        eigenvalue_ratio_test,
        full_rmt_analysis,
        test_against_mp,
        compare_matrices_rmt,
    )


def test_import_alignment_alternative_metrics():
    from alignment.alternative_metrics import (
        rv_coefficient,
        distance_correlation,
        bootstrap_metric,
        permutation_test,
        cca_correlation,
        pls_score,
    )


def test_import_alignment_congruence():
    from alignment.congruence import CongruenceCoefficient, AlignmentTester


def test_import_alignment_procrustes():
    from alignment.procrustes import ProcrustesAlignment


def test_import_market_summary_statistics():
    from market.summary_statistics import SummaryStatistics, STATISTICS
    assert len(STATISTICS) == 7


def test_import_src_package():
    import src
    assert hasattr(src, "__version__")
