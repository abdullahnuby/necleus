from __future__ import annotations
import json, hashlib, time, random, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pulse_core.sequential_stdp import (
    DynamicPulseReservoir, RewardSTDPReadout, make_dataset, make_disjoint_dataset, make_jittered_copy, rasterize,
    example_raster_features, task_label, DURATION,
)

SEEDS = [104, 887, 2027, 90210, 731]
N_TRAIN_PER_TASK = 200
N_TEST_PER_TASK = 160
EPOCHS_PER_TASK = 8
BATCH_SIZE_MLP = 16
N_HIDDEN = 24
LEARNING_RATE_MLP = 0.006
PULSE_LR = 0.01
PULSE_TEMPERATURE = 0.70
TRAIN_PASSES_PULSE = EPOCHS_PER_TASK


class TemporalMLP(nn.Module):
    def __init__(self, n_in=4*DURATION, width=48, n_out=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, n_out))
    def forward(self, x):
        return self.net(x)


def seed_torch(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def train_mlp_phase(model, trials, epochs, seed, task):
    xs = np.stack([example_raster_features(t) for t in trials])
    ys = np.asarray([t.target_class for t in trials], dtype=np.int64)
    ds = TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=BATCH_SIZE_MLP, shuffle=True, generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_MLP)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    losses = []
    for _ in range(epochs):
        total, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb); n += len(xb)
        losses.append(total / max(n,1))
    return losses


def predict_mlp(model, trials):
    xs = np.stack([example_raster_features(t) for t in trials])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(xs))
        preds = logits.argmax(dim=1).cpu().numpy()
    return preds


def eval_pulse(reservoir, readout, trials):
    preds, diagnostics = [], []
    for t in trials:
        spikes = reservoir.simulate(t)
        pred, scores, post_time, virtual = readout.predict(spikes)
        preds.append(pred)
        diagnostics.append({"prediction": int(pred), "target": int(t.target_class),
                            "correct": int(pred == t.target_class), "post_time": int(post_time),
                            "virtual_decision": int(virtual), "hidden_spike_count": int(spikes.sum()),
                            "scores": [float(x) for x in scores]})
    return np.asarray(preds), diagnostics


def accuracy(preds, trials):
    ys = np.asarray([t.target_class for t in trials], dtype=np.int64)
    return float(np.mean(preds == ys))


def macro_task_classification(preds, trials):
    # Return positive/negative class accuracies as well as whole accuracy.
    ys = np.asarray([t.target_class for t in trials], dtype=np.int64)
    out = {"accuracy": float(np.mean(preds == ys))}
    for cls in sorted(set(ys.tolist())):
        mask = ys == cls
        out[f"class_{cls}_recall"] = float(np.mean(preds[mask] == cls)) if np.any(mask) else None
    return out


def run_seed(seed: int, outdir: Path):
    run_start = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    # Make the training data identical for both models and ensure pulse/MLP see
    # the same number of example presentations in task order.
    train_sync = make_disjoint_dataset(seed + 11, "sync", N_TRAIN_PER_TASK, "train")
    train_order = make_disjoint_dataset(seed + 23, "order", N_TRAIN_PER_TASK, "train")
    train_sync_keys = {(t.a, t.b) for t in train_sync}
    train_order_keys = {(t.a, t.b) for t in train_order}
    test_sync = make_disjoint_dataset(seed + 1011, "sync", N_TEST_PER_TASK, "test", forbidden=train_sync_keys)
    test_order = make_disjoint_dataset(seed + 1023, "order", N_TEST_PER_TASK, "test", forbidden=train_order_keys)
    sync_forbidden = train_sync_keys | {(t.a, t.b) for t in test_sync}
    order_forbidden = train_order_keys | {(t.a, t.b) for t in test_order}
    shifted_sync = make_disjoint_dataset(seed + 1511, "sync", N_TEST_PER_TASK, "shifted_test", forbidden=sync_forbidden)
    shifted_order = make_disjoint_dataset(seed + 1523, "order", N_TEST_PER_TASK, "shifted_test", forbidden=order_forbidden)
    jitter_sync = make_jittered_copy(seed + 2011, test_sync, jitter=1, forbidden=train_sync_keys)
    jitter_order = make_jittered_copy(seed + 2023, test_order, jitter=1, forbidden=train_order_keys)
    overlap_sync = len(train_sync_keys & {(t.a, t.b) for t in test_sync})
    overlap_order = len(train_order_keys & {(t.a, t.b) for t in test_order})
    jitter_overlap_sync = len(train_sync_keys & {(t.a, t.b) for t in jitter_sync})
    jitter_overlap_order = len(train_order_keys & {(t.a, t.b) for t in jitter_order})
    if overlap_sync or overlap_order or jitter_overlap_sync or jitter_overlap_order:
        raise RuntimeError(f"Train/test event-sequence overlap: sync={overlap_sync}, order={overlap_order}, jitter_sync={jitter_overlap_sync}, jitter_order={jitter_overlap_order}")

    # Fixed, randomly initialized recurrent LIF reservoir; same neurons persist
    # through both tasks. Only readout synapses are plastic in this v0.28 step.
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    pulse = RewardSTDPReadout(seed=seed + 30001, n_hidden=N_HIDDEN, lr=PULSE_LR, temperature=PULSE_TEMPERATURE)
    pulse_train_rng = np.random.default_rng(seed + 40001)

    # Task 1: coincidence. Precompute dynamic spike traces once per example,
    # then reuse exactly these features for the configured presentation budget.
    sync_spikes = [reservoir.simulate(t) for t in train_sync]
    pulse_rewards_sync = []
    for ep in range(TRAIN_PASSES_PULSE):
        order = pulse_train_rng.permutation(len(train_sync))
        for ix in order:
            result = pulse.train_trial(sync_spikes[int(ix)], train_sync[int(ix)].target_class, pulse_train_rng)
            pulse_rewards_sync.append(result["correct"])

    pulse_sync_before, pulse_diag_sync_before = eval_pulse(reservoir, pulse, test_sync)
    seed_torch(seed + 50001)
    mlp = TemporalMLP()
    mlp_losses_sync = train_mlp_phase(mlp, train_sync, EPOCHS_PER_TASK, seed + 60001, "sync")
    mlp_sync_before = predict_mlp(mlp, test_sync)

    # Task 2: A-before-B. Continue on the same recurrent reservoir and same
    # output weights; no replay of task-1 examples in this main condition.
    order_spikes = [reservoir.simulate(t) for t in train_order]
    pulse_rewards_order = []
    for ep in range(TRAIN_PASSES_PULSE):
        order_ix = pulse_train_rng.permutation(len(train_order))
        for ix in order_ix:
            result = pulse.train_trial(order_spikes[int(ix)], train_order[int(ix)].target_class, pulse_train_rng)
            pulse_rewards_order.append(result["correct"])

    pulse_sync_after, pulse_diag_sync_after = eval_pulse(reservoir, pulse, test_sync)
    pulse_order_after, pulse_diag_order = eval_pulse(reservoir, pulse, test_order)
    pulse_sync_jitter, pulse_diag_sync_jitter = eval_pulse(reservoir, pulse, jitter_sync)
    pulse_order_jitter, pulse_diag_order_jitter = eval_pulse(reservoir, pulse, jitter_order)
    pulse_sync_shifted, pulse_diag_sync_shifted = eval_pulse(reservoir, pulse, shifted_sync)
    pulse_order_shifted, pulse_diag_order_shifted = eval_pulse(reservoir, pulse, shifted_order)

    mlp_losses_order = train_mlp_phase(mlp, train_order, EPOCHS_PER_TASK, seed + 70001, "order")
    mlp_sync_after = predict_mlp(mlp, test_sync)
    mlp_order_after = predict_mlp(mlp, test_order)
    mlp_sync_jitter = predict_mlp(mlp, jitter_sync)
    mlp_order_jitter = predict_mlp(mlp, jitter_order)
    mlp_sync_shifted = predict_mlp(mlp, shifted_sync)
    mlp_order_shifted = predict_mlp(mlp, shifted_order)

    # Check dynamic state/noise: compare pulse traces on original vs jittered
    # examples with labels computed from the actual jittered input sequence.
    pulse_stats = {
        "seed": seed,
        "data_integrity": {"exact_train_test_overlap_sync": overlap_sync,
                            "exact_train_test_overlap_order": overlap_order,
                            "exact_jitter_train_overlap_sync": jitter_overlap_sync,
                            "exact_jitter_train_overlap_order": jitter_overlap_order,
                            "jitter_changed_fraction_sync": float(np.mean([(a.a,a.b)!=(b.a,b.b) for a,b in zip(test_sync,jitter_sync)])),
                            "jitter_changed_fraction_order": float(np.mean([(a.a,a.b)!=(b.a,b.b) for a,b in zip(test_order,jitter_order)])),
                            "jitter_label_preservation_sync": float(np.mean([a.label == b.label for a,b in zip(test_sync, jitter_sync)])),
                            "jitter_label_preservation_order": float(np.mean([a.label == b.label for a,b in zip(test_order, jitter_order)]))},
        "pulse": {
            "task1_sync_before_task2": macro_task_classification(pulse_sync_before, test_sync),
            "task1_sync_after_task2": macro_task_classification(pulse_sync_after, test_sync),
            "task2_order_after_training": macro_task_classification(pulse_order_after, test_order),
            "task1_sync_jittered": macro_task_classification(pulse_sync_jitter, jitter_sync),
            "task2_order_jittered": macro_task_classification(pulse_order_jitter, jitter_order),
            "task1_sync_shifted_phase": macro_task_classification(pulse_sync_shifted, shifted_sync),
            "task2_order_shifted_phase": macro_task_classification(pulse_order_shifted, shifted_order),
            "forgetting_task1_pp": 100.0 * (accuracy(pulse_sync_before, test_sync) - accuracy(pulse_sync_after, test_sync)),
            "reward_accuracy_task1_training": float(np.mean(pulse_rewards_sync)),
            "reward_accuracy_task2_training": float(np.mean(pulse_rewards_order)),
            "weight_min": float(pulse.w.min()), "weight_max": float(pulse.w.max()),
            "weight_mean": float(pulse.w.mean()),
            "virtual_decision_fraction_task1_test": float(np.mean([d["virtual_decision"] for d in pulse_diag_sync_after])),
            "virtual_decision_fraction_task2_test": float(np.mean([d["virtual_decision"] for d in pulse_diag_order])),
            "hidden_spikes_mean_task1": float(np.mean([d["hidden_spike_count"] for d in pulse_diag_sync_after])),
            "hidden_spikes_mean_task2": float(np.mean([d["hidden_spike_count"] for d in pulse_diag_order])),
        },
        "mlp": {
            "task1_sync_before_task2": macro_task_classification(mlp_sync_before, test_sync),
            "task1_sync_after_task2": macro_task_classification(mlp_sync_after, test_sync),
            "task2_order_after_training": macro_task_classification(mlp_order_after, test_order),
            "task1_sync_jittered": macro_task_classification(mlp_sync_jitter, jitter_sync),
            "task2_order_jittered": macro_task_classification(mlp_order_jitter, jitter_order),
            "task1_sync_shifted_phase": macro_task_classification(mlp_sync_shifted, shifted_sync),
            "task2_order_shifted_phase": macro_task_classification(mlp_order_shifted, shifted_order),
            "forgetting_task1_pp": 100.0 * (accuracy(mlp_sync_before, test_sync) - accuracy(mlp_sync_after, test_sync)),
            "task1_final_train_loss": float(mlp_losses_sync[-1]),
            "task2_final_train_loss": float(mlp_losses_order[-1]),
        },
    }

    # Save light checkpoints/diagnostics for reproducibility.
    np.savez_compressed(outdir / f"seed_{seed}_pulse_state.npz", w_in=reservoir.w_in, w_rec=reservoir.w_rec,
                        w_out=pulse.w, tau_mem=reservoir.tau_mem, tau_syn=reservoir.tau_syn,
                        thresholds=reservoir.threshold)
    pulse_stats["elapsed_seconds"] = float(time.time() - run_start)
    with (outdir / f"seed_{seed}_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": pulse_stats, "pulse_task1_after": pulse_diag_sync_after,
                   "pulse_task2_after": pulse_diag_order}, f, ensure_ascii=False, indent=2)
    return pulse_stats


def aggregate(rows):
    methods = ("pulse", "mlp")
    paths = [
        ("task1_sync_before_task2", "accuracy"),
        ("task1_sync_after_task2", "accuracy"),
        ("task2_order_after_training", "accuracy"),
        ("task1_sync_jittered", "accuracy"),
        ("task2_order_jittered", "accuracy"),
        ("task1_sync_shifted_phase", "accuracy"),
        ("task2_order_shifted_phase", "accuracy"),
        ("forgetting_task1_pp", None),
    ]
    res = {}
    for method in methods:
        res[method] = {}
        for name, key in paths:
            vals = [r[method][name][key] if key else r[method][name] for r in rows]
            res[method][name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                 "min": float(np.min(vals)), "max": float(np.max(vals)), "per_seed": vals}
    for field in ("reward_accuracy_task1_training", "reward_accuracy_task2_training", "virtual_decision_fraction_task1_test", "virtual_decision_fraction_task2_test", "hidden_spikes_mean_task1", "hidden_spikes_mean_task2"):
        vals = [r["pulse"][field] for r in rows]
        res["pulse"][field] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "per_seed": vals}
    return res


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_preregistration() -> dict:
    path = ROOT / "results" / "sequential_stdp_v0_28_1" / "preregistration.json"
    if not path.exists():
        raise RuntimeError("Missing preregistration.json; refusing to run final benchmark without a frozen protocol")
    data = json.loads(path.read_text(encoding="utf-8"))
    for rel, expected in data["sha256"].items():
        actual = sha256_file(ROOT / rel)
        if actual != expected:
            raise RuntimeError(f"Protocol code hash mismatch for {rel}: expected {expected}, got {actual}")
    return data


def main():
    prereg = verify_preregistration()
    start = time.time()
    outdir = ROOT / "results" / "sequential_stdp_v0_28_1"
    outdir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for seed in SEEDS:
        t0 = time.time()
        row = run_seed(seed, outdir)
        all_rows.append(row)
        # Store after each seed so no progress is lost if an environment limit hits.
        temp = {"version": "0.28.1", "seeds": SEEDS[:len(all_rows)], "per_seed": all_rows, "aggregate": aggregate(all_rows)}
        (outdir / "results_partial.json").write_text(json.dumps(temp, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"seed={seed} pulse sync {row['pulse']['task1_sync_before_task2']['accuracy']:.3f}->{row['pulse']['task1_sync_after_task2']['accuracy']:.3f}; order={row['pulse']['task2_order_after_training']['accuracy']:.3f}; MLP sync {row['mlp']['task1_sync_before_task2']['accuracy']:.3f}->{row['mlp']['task1_sync_after_task2']['accuracy']:.3f}; MLP order={row['mlp']['task2_order_after_training']['accuracy']:.3f}; {time.time()-t0:.1f}s", flush=True)
    results = {"version": "0.28.1", "seeds": SEEDS, "per_seed": all_rows, "aggregate": aggregate(all_rows),
               "runtime_seconds": time.time()-start,
               "protocol": {"train_per_task": N_TRAIN_PER_TASK, "test_per_task": N_TEST_PER_TASK,
                            "epochs_per_task": EPOCHS_PER_TASK, "duration": DURATION, "n_hidden": N_HIDDEN,
                            "mlp_batch_size": BATCH_SIZE_MLP, "mlp_learning_rate": LEARNING_RATE_MLP,
                            "pulse_lr": PULSE_LR, "pulse_temperature": PULSE_TEMPERATURE, "output_classes": 2,
                            "preregistration_sha256": sha256_file(outdir / "preregistration.json")}}
    (outdir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "results_partial.json").unlink(missing_ok=True)
    print(json.dumps(results["aggregate"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
