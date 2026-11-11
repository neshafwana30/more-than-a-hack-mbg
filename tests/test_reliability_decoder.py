import unittest
import numpy as np
from src.reliability_viterbi import reliability_viterbi
from src.temporal_decoder import viterbi

CFG = {
    "home_away_switch_penalty": 5.0,
    "play_dead_switch_penalty": 3.3,
    "unreliable_team_penalty_boost": 0.5,
    "unreliable_dead_penalty_boost": 0.0,
}


class TestReliabilityViterbi(unittest.TestCase):
    def test_all_reliable_matches_baseline(self):
        rng = np.random.default_rng(2027)
        for n in [1, 10, 100, 1000]:
            p = rng.dirichlet([1.0, 1.0, 1.0], size=n)
            self.assertTrue(np.array_equal(
                reliability_viterbi(p, CFG, np.ones(n)), viterbi(p, CFG)
            ))

    def test_boost_zero_matches_baseline_even_if_missing(self):
        rng = np.random.default_rng(17)
        p = rng.dirichlet([1.0, 1.0, 1.0], size=100)
        r = rng.random(100)
        cfg = {**CFG, "unreliable_team_penalty_boost": 0.0}
        self.assertTrue(np.array_equal(
            reliability_viterbi(p, cfg, r), viterbi(p, cfg)
        ))

    def test_invalid_reliability_raises(self):
        p = np.full((3, 3), 1/3)
        with self.assertRaises(ValueError):
            reliability_viterbi(p, CFG, np.ones(2))
        with self.assertRaises(ValueError):
            reliability_viterbi(p, CFG, np.array([1.0, -0.1, 0.5]))

    def test_empty(self):
        self.assertEqual(len(reliability_viterbi(np.empty((0, 3)), CFG, np.empty(0))), 0)


if __name__ == "__main__":
    unittest.main()
