import unittest
from pulse_core.model import LIFNeuron, Synapse, PulseNetwork, unit_impulse_eligibility
from pulse_core.learning import build_network, predict, make_train_set, make_test_set, train_one_seed

class PulseModelTests(unittest.TestCase):
    def test_single_input_pulse_transmits_after_delay(self):
        net = PulseNetwork([LIFNeuron("n", threshold=10.0)], [Synapse("A_n", "A", "n", 0.7, delay=2)])
        result = net.run({"A": [3]}, duration=10)
        self.assertEqual(result.delivered_pulses[0][0], 5)
        self.assertAlmostEqual(result.delivered_pulses[0][3], 0.7)

    def test_no_pulse_no_current(self):
        net = PulseNetwork([LIFNeuron("n")], [Synapse("A_n", "A", "n", 0.7)])
        result = net.run({"A": []}, duration=10)
        self.assertEqual(result.spike_times["n"], [])
        self.assertTrue(all(v == 0.0 for v in result.current_trace["n"]))

    def test_threshold_emits_spike_and_reset(self):
        net = PulseNetwork([LIFNeuron("n", tau_mem=1.0, tau_syn=100.0, threshold=0.5, refractory_steps=2)],
                           [Synapse("A_n", "A", "n", 3.0, delay=0)])
        result = net.run({"A": [0]}, duration=8)
        self.assertTrue(result.spike_times["n"])
        self.assertEqual(result.spike_times["n"][0], 0)

    def test_spikes_propagate_through_two_synapses(self):
        ns = [LIFNeuron("hidden", threshold=0.5, tau_mem=1, tau_syn=20), LIFNeuron("out", threshold=0.5, tau_mem=1, tau_syn=20)]
        ss = [Synapse("A_h", "A", "hidden", 3, delay=0), Synapse("h_o", "hidden", "out", 3, delay=1)]
        res = PulseNetwork(ns, ss).run({"A": [1]}, 10)
        self.assertTrue(res.spike_times["hidden"])
        self.assertTrue(res.spike_times["out"])
        self.assertGreaterEqual(res.spike_times["out"][0], res.spike_times["hidden"][0] + 1)

    def test_subthreshold_eligibility_is_zero_without_input(self):
        e = unit_impulse_eligibility({"A": [], "B": []}, 10, 2, 6, {"A": 1, "B": 1})
        self.assertEqual(e["A"], 0)
        self.assertEqual(e["B"], 0)

    def test_train_test_start_ranges_disjoint(self):
        tr = make_train_set(5)
        te = make_test_set(5)
        self.assertTrue(all(max(ex.a + ex.b) <= 17 for ex in tr))
        self.assertTrue(all(min(ex.a + ex.b) >= 9 for ex in te))

    def test_dataset_has_all_three_pattern_kinds(self):
        kinds = {x.kind for x in make_train_set(2)}
        self.assertEqual(kinds, {"coincident", "separated", "A_only", "B_only"})

    def test_training_changes_weights(self):
        result = train_one_seed(123, epochs=2)
        self.assertEqual(len(result["final_weights"]), 2)
        self.assertTrue(all(0.05 <= w <= 2.5 for w in result["final_weights"]))

    def test_inference_emits_pulses_in_chain(self):
        net = build_network((1.1, 1.1))
        prediction, result = predict(net, [5], [5], duration=28)
        self.assertEqual(prediction, 1)
        self.assertTrue(result.spike_times["coincidence"])
        self.assertTrue(result.spike_times["readout"])

    def test_temporal_separation_changes_response(self):
        net = build_network((1.0, 1.0))
        synchronous, _ = predict(net, [5], [5], duration=28)
        separated, _ = predict(net, [5], [11], duration=28)
        self.assertEqual(synchronous, 1)
        self.assertEqual(separated, 0)

if __name__ == "__main__":
    unittest.main()
