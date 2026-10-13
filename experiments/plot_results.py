from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制性能对照图")
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmark_latency.png"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.summary_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("汇总 CSV 为空")
    import matplotlib.pyplot as plt

    labels = [row["variant"] for row in rows]
    means = [float(row["algorithm_total_ms_mean"]) for row in rows]
    p95 = [float(row["algorithm_total_ms_p95"]) for row in rows]
    figure, axis = plt.subplots(figsize=(9, 5.2), layout="constrained")
    x = list(range(len(labels)))
    axis.bar(x, means, label="Mean", color="#4c78a8")
    axis.plot(x, p95, marker="o", color="#e45756", label="P95")
    axis.set_xticks(x, labels, rotation=15, ha="right")
    axis.set_ylabel("Algorithm latency (ms)")
    axis.set_title("Face comparison latency by variant")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
