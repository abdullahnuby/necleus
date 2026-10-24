import unittest
from pulse_core.model import LIFNeuron, Synapse, PulseNetwork
from pulse_core.tasks import make_dataset, coincidence_rule
from pulse_core.reference import reference_lif_trace


def build_and():
    n = LIFNeuron("and", tau_mem=1.0, tau_syn=0.8, threshold=1.02,
                   refractory_steps=2, clear_synaptic_current_on_spike=True)
    ss = [
        Synapse("A_to_and", "A", "and", 1.0, delay=1, tau_syn=0.8, current_cap=1.0),
        Synapse("B_to_and", "B", "and", 1.0, delay=1, tau_syn=0.8, current_cap=1.0),
    ]
    return PulseNetwork([n], ss)


class SaturatedAndTests(unittest.TestCase):
    def test_simultaneous_inputs_fire(self):
        r = build_and().run({"A": [5], "B": [5]}, 32)
        self.assertTrue(r.spike_times["and"])

    def test_near_inputs_in_both_orders_fire(self):
        for a, b in [([5], [6]), ([6], [5])]:
            r = build_and().run({"A": a, "B": b}, 32)
            self.assertTrue(r.spike_times["and"])

    def test_lag_two_does_not_fire(self):
        r = build_and().run({"A": [5], "B": [7]}, 32)
        self.assertFalse(r.spike_times["and"])

    def test_single_channel_high_rate_burst_does_not_fire(self):
        r = build_and().run({"A": list(range(2, 27)), "B": []}, 32)
        self.assertFalse(r.spike_times["and"])
        self.assertLessEqual(max(abs(x) for x in r.synapse_current_trace["A_to_and"]), 1.0 + 1e-12)

    def test_repeated_separated_trains_do_not_fire(self):
        r = build_and().run({"A": [2, 8, 14, 20], "B": [4, 10, 16, 22]}, 32)
        self.assertFalse(r.spike_times["and"])

    def test_synchronous_trains_fire(self):
        r = build_and().run({"A": [2, 8, 14, 20], "B": [2, 8, 14, 20]}, 32)
        self.assertTrue(r.spike_times["and"])

    def test_inhibitory_synapse_reduces_response(self):
        exc = PulseNetwork([LIFNeuron("n", tau_mem=1, tau_syn=1, threshold=0.8)],
                           [Synapse("exc", "A", "n", 1.5, delay=0, tau_syn=1)])
        inh = PulseNetwork([LIFNeuron("n", tau_mem=1, tau_syn=1, threshold=0.8)], [
            Synapse("exc", "A", "n", 1.5, delay=0, tau_syn=1),
            Synapse("inh", "B", "n", 0.8, delay=0, polarity=-1, tau_syn=1),
        ])
        r_exc = exc.run({"A": [2]}, 10)
        r_inh = inh.run({"A": [2], "B": [2]}, 10)
        self.assertTrue(r_exc.spike_times["n"])
        self.assertFalse(r_inh.spike_times["n"])

    def test_neuronal_zero_delay_is_still_causal(self):
        net = PulseNetwork(
            [LIFNeuron("a", tau_mem=1, tau_syn=10, threshold=0.5), LIFNeuron("b", tau_mem=1, tau_syn=10, threshold=0.5)],
            [Synapse("src_a", "SRC", "a", 3, delay=0), Synapse("a_b", "a", "b", 3, delay=0)],
        )
        r = net.run({"SRC": [1]}, 8)
        self.assertTrue(r.spike_times["a"])
        self.assertTrue(r.spike_times["b"])
        self.assertGreater(r.spike_times["b"][0], r.spike_times["a"][0])

    def test_seeded_varied_test_distribution_is_consistent_with_task_rule(self):
        for seed in range(10):
            for ex in make_dataset(seed, n=100):
                self.assertEqual(ex.target, coincidence_rule(ex.a, ex.b, 1))

    def test_voltage_trace_records_threshold_crossing_before_reset(self):
        net = PulseNetwork([LIFNeuron("n", tau_mem=1.0, tau_syn=10.0, threshold=0.8)],
                           [Synapse("src", "A", "n", 1.5, delay=0, tau_syn=10.0)])
        r = net.run({"A": [2]}, duration=8)
        self.assertTrue(r.spike_times["n"])
        first = r.spike_times["n"][0]
        self.assertGreaterEqual(r.voltage_trace["n"][first], 0.8)
        if first + 1 < len(r.voltage_trace["n"]):
            self.assertAlmostEqual(r.voltage_trace["n"][first + 1], 0.0, delta=1e-12)

    def test_reference_recurrence_matches_event_driven_model(self):
        times = [2, 5, 10, 11]
        tau_m, tau_s, cap = 2.3, 1.7, 0.9
        net = PulseNetwork([LIFNeuron("n", tau_mem=tau_m, tau_syn=tau_s, threshold=100)],
                           [Synapse("src", "SRC", "n", 1, delay=0, tau_syn=tau_s, current_cap=cap)])
        got = net.run({"SRC": times}, 20)
        ri, rv, rs = reference_lif_trace(20, [(t, 1.0) for t in times], tau_m, tau_s, 100, current_cap=cap)
        self.assertEqual(rs, [])
        self.assertLess(max(abs(a-b) for a,b in zip(got.current_trace["n"], ri)), 1e-12)
        self.assertLess(max(abs(a-b) for a,b in zip(got.voltage_trace["n"], rv)), 1e-12)

if __name__ == "__main__":
    unittest.main()
