import unittest

from src.rs_cusum_phase2r.protocol import load_raw_config, stable_seed


class Phase2RProtocolTests(unittest.TestCase):
    def test_protocol_is_n0_only_and_gate_closed(self):
        cfg = load_raw_config()
        self.assertEqual(cfg["dgp"]["scenario"], "N0_complete_balanced_null")
        self.assertFalse(cfg["protocol"]["alternatives_allowed"])
        self.assertFalse(cfg["protocol"]["method_revision_allowed"])
        self.assertEqual(cfg["protocol"]["formal_ohiot1dm_gate"], "CLOSED")

    def test_required_repetition_floors(self):
        cfg = load_raw_config()
        self.assertGreaterEqual(cfg["decomposition"]["truth_replicates"], 1000)
        self.assertGreaterEqual(cfg["n_scaling"]["replicates"], 300)
        self.assertEqual(cfg["bootstrap_forensics"]["datasets"], 50)
        self.assertGreaterEqual(cfg["bootstrap_forensics"]["draws"], 1999)

    def test_stable_seed(self):
        self.assertEqual(stable_seed("a", 1), stable_seed("a", 1))
        self.assertNotEqual(stable_seed("a", 1), stable_seed("a", 2))


if __name__ == "__main__":
    unittest.main()
