"""CLI regression checks; run with Python 3 and the built demo executable path."""

import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run_demo(executable, directory, name, mode="cicfm", strategy="adaptive", interval=2):
    path = directory / (name + ".csv")
    subprocess.run(
        [str(executable), "--mode", mode, "--sampling", strategy,
         "--points", "16", "--samples", "8", "--steps", "3", "--hidden", "4",
         "--frequencies", "2", "--eval-every", str(interval), "--eval-probes", "2",
         "--csv", str(path)], check=True, capture_output=True, text=True,
    )
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 4, "one initial row plus three training rows")
    previous_coverage = 0
    stage_names = ["sampling_ms", "gather_ms", "embedding_ms", "train_ms",
                   "residual_ms", "update_ms"]
    for step, row in enumerate(rows):
        require(None not in row and None not in row.values(), "CSV column alignment")
        require(int(row["step"]) == step, "step order")
        require(row["mode"] == mode and row["sampling"] == strategy, "run metadata")
        require(row["all_finite"] == "1", "finite diagnostics")
        require(row["eval_metric"] == ("rgb_mse" if mode == "mse" else "velocity_mse"),
                "evaluation label")
        require(int(row["eval_probes"]) == (1 if mode == "mse" else 2), "probe count")
        require(bool(row["eval_loss"]) == (step == 0 or step % interval == 0 or step == 3),
                "evaluation cadence")
        for key, value in row.items():
            if key not in {"mode", "sampling", "eval_metric", "build_type"} and value:
                require(math.isfinite(float(value)), "non-finite CSV value: " + key)
        coverage = int(row["coverage_count"])
        require(previous_coverage <= coverage <= min(16, step * 8), "cumulative coverage")
        require(float(row["coverage_fraction"]) == coverage / 16, "coverage fraction")
        previous_coverage = coverage
        unique = int(row["unique_points"])
        require((1 <= unique <= 8) if step else unique == 0, "per-step unique count")
        expected_duplicates = 1 - unique / 8 if step else 0
        require(float(row["duplicate_rate"]) == expected_duplicates, "duplicate fraction")
        if step:
            require(float(row["sampled_loss"]) >= 0, "training loss")
            require(int(row["step_noise_seed"]) == 123 + 2 * (step - 1), "noise seed schedule")
        else:
            require(row["sampled_loss"] == row["step_noise_seed"] == "", "initial row")
        if row["eval_loss"]:
            require(float(row["eval_loss"]) >= 0, "evaluation loss")
        require(0 < float(row["probability_min"]) <= float(row["probability_max"]) <= 1,
                "probability range")
        require(float(row["probability_min"]) >= float(row["eta"]) / 16 - 1e-12,
                "uniform exploration floor")
        require(0 <= float(row["priority_min"]) <= float(row["priority_max"]),
                "priority range")
        for field in stage_names + ["step_ms", "eval_ms"]:
            require(float(row[field]) >= 0, "negative timing")
        require(sum(float(row[key]) for key in stage_names) <= float(row["step_ms"]) + 1e-9,
                "stage timings exceed total")
        if strategy == "uniform":
            require(float(row["probability_min"]) == float(row["probability_max"]) == 1 / 16,
                    "uniform probabilities")
            require(float(row["priority_min"]) == float(row["priority_max"]) == 1,
                    "uniform priorities unchanged")
            require(float(row["update_ms"]) == 0, "uniform updates skipped")
    if strategy == "adaptive":
        require(float(rows[-1]["probability_min"]) < float(rows[-1]["probability_max"]),
                "adaptive feedback must change the next distribution")
    return rows


def without_timing(row):
    return {key: value for key, value in row.items() if not key.endswith("_ms")}


def main():
    executable = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="tinyinr-demo-test-") as tmp:
        directory = Path(tmp)
        for mode in ("mse", "cicfm"):
            runs = {}
            for strategy in ("uniform", "adaptive"):
                runs[strategy] = run_demo(executable, directory, mode + strategy, mode, strategy)
            require(runs["uniform"][0]["eval_loss"] == runs["adaptive"][0]["eval_loss"],
                    "strategies must start from the same model and evaluation probes")
        repeated = run_demo(executable, directory, "repeat")
        require([without_timing(row) for row in repeated] ==
                [without_timing(row) for row in runs["adaptive"]], "seeded replay")
        frequent = run_demo(executable, directory, "frequent", interval=1)
        for a, b in zip(repeated, frequent):
            comparable = set(a) - {key for key in a if key.endswith("_ms")}
            comparable -= {"eval_every", "eval_loss"}
            require(all(a[key] == b[key] for key in comparable),
                    "evaluation must not change training or RNG state")
            if a["eval_loss"]:
                require(a["eval_loss"] == b["eval_loss"], "fixed evaluation probes")

        # Smallest and incomplete-grid images; eta=1 must match the uniform path.
        for mode in ("mse", "cicfm"):
            for points in (1, 7):
                edge_runs = []
                for strategy in ("uniform", "adaptive"):
                    path = directory / "edge.csv"
                    subprocess.run(
                        [str(executable), "--mode", mode, "--sampling", strategy,
                         "--eta", "1", "--beta", "0", "--points", str(points),
                         "--samples", str(points), "--steps", "1", "--hidden", "1",
                         "--frequencies", "1", "--eval-every", "10", "--eval-probes", "1",
                         "--noise-seed", "4294967294", "--eval-seed", "4294967294",
                         "--csv", str(path)], check=True, capture_output=True, text=True,
                    )
                    with path.open(newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    require(len(rows) == 2 and all(row["eval_loss"] for row in rows),
                            "initial and final evaluation even beyond the interval")
                    for row in rows:
                        require(math.isfinite(float(row["eval_loss"])), "finite edge evaluation")
                        require(row["priority_min"] == row["priority_max"] == "1",
                                "eta=1 skips updates, including beta=0")
                    if points == 1:
                        require(rows[-1]["coverage_count"] == "1" and
                                rows[-1]["duplicate_rate"] == "0", "single-point coverage")
                    edge_runs.append([{key: value for key, value in without_timing(row).items()
                                       if key != "sampling"} for row in rows])
                require(edge_runs[0] == edge_runs[1], "adaptive eta=1 equals uniform")

        sentinel = directory / "invalid.csv"
        sentinel.write_text("keep this file")
        invalid_options = [
            ["--mode", "bad"], ["--sampling", "bad"], ["--points", "0"],
            ["--samples", "999"], ["--steps", "-1"], ["--steps", "2junk"],
            ["--steps", "0"], ["--hidden", "0"], ["--frequencies", "2147483647"],
            ["--learning-rate", "nan"], ["--learning-rate", "0"], ["--eta", "0"],
            ["--beta", "1"], ["--eval-every", "0"], ["--eval-probes", "0"],
            ["--noise-seed", "4294967295"], ["--eval-seed", "4294967295"],
            ["--model-seed", "4294967296"], ["--unknown", "1"],
            ["--steps", "1", "--steps", "2"], ["--steps"],
        ]
        for options in invalid_options:
            result = subprocess.run([str(executable), "--csv", str(sentinel)] + options,
                                    capture_output=True, text=True)
            require(result.returncode != 0 and result.stderr, "invalid arguments must fail")
            require(sentinel.read_text() == "keep this file", "invalid args must not truncate CSV")
        result = subprocess.run([str(executable), "--points", "1", "--samples", "1",
                                 "--steps", "1", "--hidden", "1", "--frequencies", "1",
                                 "--csv", str(directory / "missing" / "output.csv")],
                                capture_output=True, text=True)
        require(result.returncode != 0, "unwritable output path must fail")
        subprocess.run([str(executable), "--help"], check=True, capture_output=True)
    print("test_adaptive_sampling_demo PASS")


if __name__ == "__main__":
    main()
