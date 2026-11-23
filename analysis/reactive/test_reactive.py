"""Regression tests for observation logic, missingness, pairing and RNG preservation."""
import unittest
import numpy as np
from run_reactive import trigger, cluster_intervals, evaluate_counts, noisy_measurements
from abc_core import score_at_switch


class PolicyTests(unittest.TestCase):
    def test_nondetect_preserves_equality_and_missing(self):
        from run_reactive import mask_ned
        x = np.log10(1 + np.array([9999., 10000., 10001., np.nan]))
        np.testing.assert_equal(mask_ned(x), [0., x[1], x[2], np.nan])

    def test_mask_precedes_slopes_and_references_are_also_masked(self):
        from unittest.mock import patch
        from run_reactive import ned_forecasts
        ref = np.log10(1 + np.array([[20000, 15000, 10000, 9999], [20000, 14500, 9999, 8000]]))
        data = {'ref_x': ref, 'ref_y': np.array([5, 7]), 'surface': np.full((2,20), .5)}
        with patch('run_reactive.abc_noisy', return_value=np.array([5., 7.])) as abc:
            ned_forecasts(data, ref)
        masked_ref, _, masked_query = abc.call_args.args
        expected = ref.copy()
        expected[0,3] = expected[1,2] = expected[1,3] = 0
        np.testing.assert_array_equal(masked_ref, expected)
        np.testing.assert_array_equal(masked_query, np.unique(expected, axis=0))

    def test_no_final_scan_fallback(self):
        for rule in ('first_increase', 'progression_1.20', 'ned_10000'):
            switch, _, _ = trigger(np.array([[30000, 20000, 9000, 5000]]), [0, 4, 8, 12], rule)
            self.assertTrue(np.isnan(switch[0]))

    def test_progression_uses_all_earlier_scans(self):
        x = np.array([[200, 100, 110, 125], [200, 100, 110, 119], [0, 0, 0, 0]])
        sw, _, _ = trigger(x, [0, 4, 8, 12], 'progression_1.20')
        np.testing.assert_equal(sw, [12, np.nan, np.nan])
        sw, _, _ = trigger(x, [0, 4, 8, 12], 'first_increase')
        np.testing.assert_equal(sw, [8, 8, np.nan])

    def test_ned_strict_entry_and_exit(self):
        x = np.array([[20000, 10000, 15000], [20000, 9999, 10000], [20000, 9999, 10001]])
        sw, entered, _ = trigger(x, [0, 4, 8], 'ned_10000')
        np.testing.assert_equal(sw, [np.nan, np.nan, 8])
        np.testing.assert_equal(entered, [False, True, True])

    def test_late_missing_keeps_earlier_trigger(self):
        x = np.array([[20000, 9000, 12000, np.nan], [20000, 9000, 8000, np.nan]])
        sw, entered, complete = trigger(x, [0, 4, 16, 19], 'ned_10000')
        np.testing.assert_equal(sw, [16, np.nan])
        self.assertTrue(entered.all())
        self.assertFalse(complete.any())

    def test_endpoint_and_missing_outcome_lookup(self):
        surface = np.vstack([np.arange(20), np.arange(20)]) / 20
        np.testing.assert_equal(score_at_switch(surface, np.array([19, np.nan])), [.95, np.nan])
        with self.assertRaises(ValueError):
            score_at_switch(surface, np.array([20, 0]))

    def test_matched_abc_excludes_nontriggers(self):
        data = {'ids': np.array([1, 2]), 'tmin': np.array([4., 4.]), 'surface': np.array([[.2]*20, [.8]*20])}
        row, table = evaluate_counts(data, np.array([[300, 100, 200], [300, 100, 50]]),
                                     [0, 4, 8], 'first_increase', np.array([.3, .9]), 0, 40, 2)
        self.assertEqual(row['n_triggered'], 1)
        self.assertAlmostEqual(row['abc_extinction_matched'], .3)
        self.assertAlmostEqual(row['abc_extinction_full_cohort'], .6)
        self.assertTrue(np.isnan(table.loc[1, 'reactive_extinction']))

    def test_cluster_resampling_keeps_patient_draws(self):
        hit = np.ones((1, 3), bool)
        value = np.array([[.1, .5, .9]])
        first = cluster_intervals(hit, {'effect': value}, 1000, 8)
        repeated = cluster_intervals(np.repeat(hit, 20, axis=0), {'effect': np.repeat(value, 20, axis=0)}, 1000, 8)
        self.assertAlmostEqual(first['effect_ci_low'], repeated['effect_ci_low'])
        self.assertAlmostEqual(first['effect_ci_high'], repeated['effect_ci_high'])

    def test_early_rng_matches_original_and_late_missing_stays_missing(self):
        data = {'ids': np.arange(3), 'population': np.full((3, 20), 10000.)}
        data['population'][2, 19] = np.nan
        rng = np.random.default_rng(20260930)
        historical = np.random.default_rng(20260930)
        for si, sigma in enumerate([0., .05, .1]):
            noisy, obs = noisy_measurements(data, sigma, si, rng, 7, 20260930)
            expected = np.log10(10001) + historical.normal(0, sigma, (7, 3, 3, 13))
            np.testing.assert_array_equal(noisy, expected)
            self.assertTrue(np.isnan(obs[:, 2, 19]).all())
            np.testing.assert_allclose(obs[:, :, [0, 4, 8, 12]], np.maximum(10**expected[:, :, 0, [0, 4, 8, 12]] - 1, 0))

    def test_bad_observations_rejected(self):
        for value in (-1, np.inf):
            with self.assertRaises(ValueError):
                trigger(np.array([[10, value]]), [0, 4], 'first_increase')


if __name__ == '__main__':
    unittest.main()
