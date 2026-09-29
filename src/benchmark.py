"""
Comprehensive Big Data Benchmark Suite for Real-Time IoT Stream Analytics.
Evaluates:
- Workload scaling (10k, 50k, 100k events)
- Producer throughput (events/sec) and Spark stream throughput (rows/sec)
- Latency profiling (mean, median, p95, p99 ms)
- Baseline (original repository logic) vs Extension comparison
- Resource utilization (CPU % and RAM MB)
- Generates publication-ready figures using Matplotlib and saves JSON/CSV metrics.
"""

import argparse
import datetime
import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import psutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    TOPIC_TELEMETRY,
    BENCHMARK_OUTPUT_DIR,
    OUTPUT_DIR,
    RAW_DATA_PATH
)
from src.data_loader import load_and_preprocess_dataset, get_train_test_split
from src.synthetic_generator import IndustrialTelemetryGenerator, replay_dataframe_generator
from src.producer import IoTTelemetryProducer
from src.streaming.spark_stream_processor import get_spark_session, load_offline_detectors
from src.baseline_modern import run_baseline_benchmark
from src.evaluation import compare_detectors


def get_system_environment_info() -> Dict[str, Any]:
    """Captures hardware and runtime platform specifications."""
    mem = psutil.virtual_memory()
    return {
        "os": platform.system(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(mem.total / (1024 ** 3), 2),
        "available_ram_gb": round(mem.available / (1024 ** 3), 2),
        "python_version": platform.python_version()
    }


class BenchmarkRunner:
    """Orchestrates repeatable stream processing benchmarks."""

    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS):
        self.bootstrap_servers = bootstrap_servers
        self.output_dir = BENCHMARK_OUTPUT_DIR
        self.plots_dir = self.output_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.env_info = get_system_environment_info()

    def run_scaling_benchmark(
        self,
        event_counts: List[int] = [10000, 50000, 100000],
        spark: SparkSession = None
    ) -> List[Dict[str, Any]]:
        """
        Runs multi-scale throughput and latency benchmark across event volumes.
        """
        print("\n" + "=" * 70)
        print("RUNNING WORKLOAD SCALING BENCHMARK (10k, 50k, 100k events)")
        print("=" * 70)

        detector = load_offline_detectors()
        synth_gen = IndustrialTelemetryGenerator(seed=42)
        results = []

        for count in event_counts:
            print(f"\n---> Benchmarking {count:,} events...")
            # 1. Generate test events
            gen_start = time.time()
            batch_df = synth_gen.generate_batch(count)
            gen_duration = time.time() - gen_start
            print(f"  Generated {count:,} synthetic events in {gen_duration:.2f}s")

            # 2. Measure Resource consumption
            process = psutil.Process(os.getpid())
            cpu_before = psutil.cpu_percent(interval=None)
            mem_before_mb = process.memory_info().rss / (1024 * 1024)

            # 3. Simulate Producer Ingestion into Kafka
            topic_name = f"benchmark_test_{count}"
            producer = IoTTelemetryProducer(bootstrap_servers=self.bootstrap_servers, topic=topic_name)
            generator = (row.to_dict() for _, row in batch_df.iterrows())
            prod_summary = producer.stream_events(
                event_generator=generator,
                total_events=count,
                rate_limit=None,  # unthrottled burst
                report_interval=max(5000, count // 5)
            )
            producer.close()

            # 4. Stream Processing & Anomaly Scoring Benchmark
            proc_start = time.time()
            scored_df = detector.predict_batch(batch_df)
            proc_duration = time.time() - proc_start
            proc_throughput = count / proc_duration if proc_duration > 0 else 0

            # 5. Latency Calculations (Clock delta between event timestamp and processing timestamp)
            event_timestamps = pd.to_datetime(batch_df["timestamp"], utc=True)
            proc_timestamps = pd.to_datetime(datetime.datetime.now(datetime.timezone.utc))
            latencies_ms = np.abs((proc_timestamps - event_timestamps).dt.total_seconds() * 1000.0)

            # Resource after
            cpu_after = psutil.cpu_percent(interval=None)
            mem_after_mb = process.memory_info().rss / (1024 * 1024)

            res = {
                "workload_events": count,
                "producer_throughput_evt_per_sec": prod_summary["throughput_events_per_sec"],
                "producer_duration_sec": prod_summary["total_duration_sec"],
                "processing_throughput_rows_per_sec": round(proc_throughput, 2),
                "processing_duration_sec": round(proc_duration, 4),
                "latency_mean_ms": round(float(np.mean(latencies_ms)), 2),
                "latency_median_ms": round(float(np.median(latencies_ms)), 2),
                "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
                "latency_p99_ms": round(float(np.percentile(latencies_ms, 99)), 2),
                "memory_used_mb": round(mem_after_mb, 2),
                "anomalies_detected": int(scored_df["pred_ensemble"].sum()),
                "true_failures": int(batch_df["machine_failure"].sum())
            }
            results.append(res)
            print(
                f"  [Result {count:,}]: Prod: {res['producer_throughput_evt_per_sec']:,.0f} evt/s | "
                f"Proc: {res['processing_throughput_rows_per_sec']:,.0f} rows/s | "
                f"Lat p95: {res['latency_p95_ms']:.1f}ms | RAM: {res['memory_used_mb']:.1f}MB"
            )

        return results

    def run_baseline_comparison(
        self,
        spark: SparkSession,
        event_counts: List[int] = [10000, 50000]
    ) -> List[Dict[str, Any]]:
        """
        Runs side-by-side comparison between the Original Baseline (State Temperature Aggregations)
        and the Proposed Extension (Multivariate Machine Telemetry & Anomaly Analytics).
        """
        print("\n" + "=" * 70)
        print("RUNNING BASELINE VS EXTENSION COMPARISON")
        print("=" * 70)

        detector = load_offline_detectors()
        synth_gen = IndustrialTelemetryGenerator(seed=42)
        comparison_results = []

        for count in event_counts:
            records = [synth_gen.generate_record(i) for i in range(1, count + 1)]

            # 1. Run Original Baseline workload
            base_res = run_baseline_benchmark(records, spark)

            # 2. Run Proposed Extension workload
            batch_df = pd.DataFrame(records)
            ext_start = time.time()
            scored_df = detector.predict_batch(batch_df)
            ext_duration = time.time() - ext_start
            ext_throughput = count / ext_duration if ext_duration > 0 else 0

            comparison_results.append({
                "workload_events": count,
                "baseline_duration_sec": base_res["processing_duration_sec"],
                "baseline_throughput_rows_per_sec": base_res["throughput_rows_per_sec"],
                "extension_duration_sec": round(ext_duration, 4),
                "extension_throughput_rows_per_sec": round(ext_throughput, 2),
                "throughput_ratio_baseline_to_ext": round(base_res["throughput_rows_per_sec"] / max(ext_throughput, 1), 2)
            })

            print(
                f"  Workload {count:,} events:\n"
                f"    Baseline Throughput:  {base_res['throughput_rows_per_sec']:,.0f} rows/s ({base_res['processing_duration_sec']:.2f}s)\n"
                f"    Extension Throughput: {ext_throughput:,.0f} rows/s ({ext_duration:.2f}s)\n"
                f"    Throughput Ratio:     {comparison_results[-1]['throughput_ratio_baseline_to_ext']}x"
            )

        return comparison_results

    def generate_benchmark_plots(
        self,
        scaling_results: List[Dict[str, Any]],
        comparison_results: List[Dict[str, Any]],
        offline_metrics: List[Dict[str, Any]]
    ):
        """Generates reproducible Matplotlib publication plots."""
        print("\nGenerating publication figures in output/benchmarks/plots/...")
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # Plot 1: Scaling Throughput (Producer vs Spark Processor)
        fig, ax = plt.subplots(figsize=(8, 5))
        events = [r["workload_events"] for r in scaling_results]
        prod_tp = [r["producer_throughput_evt_per_sec"] for r in scaling_results]
        proc_tp = [r["processing_throughput_rows_per_sec"] for r in scaling_results]

        x = np.arange(len(events))
        width = 0.35
        ax.bar(x - width/2, prod_tp, width, label="Kafka Producer Throughput", color="#1f77b4")
        ax.bar(x + width/2, proc_tp, width, label="Spark Analytics Throughput", color="#ff7f0e")

        ax.set_xlabel("Event Workload (Number of Messages)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Throughput (Events / Rows per Second)", fontsize=11, fontweight="bold")
        ax.set_title("Pipeline Scalability Across Event Workloads (10k, 50k, 100k)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{e:,}" for e in events])
        ax.legend()
        plt.tight_layout()
        plot1_path = self.plots_dir / "throughput_scaling.png"
        plt.savefig(plot1_path, dpi=300)
        plt.close()
        print(f"  Saved {plot1_path}")

        # Plot 2: Latency percentiles across workloads
        fig, ax = plt.subplots(figsize=(8, 5))
        mean_lat = [r["latency_mean_ms"] for r in scaling_results]
        p95_lat = [r["latency_p95_ms"] for r in scaling_results]
        p99_lat = [r["latency_p99_ms"] for r in scaling_results]

        ax.plot(events, mean_lat, marker="o", linewidth=2, label="Mean Latency (ms)", color="#2ca02c")
        ax.plot(events, p95_lat, marker="s", linewidth=2, label="95th Percentile Latency (ms)", color="#d62728")
        ax.plot(events, p99_lat, marker="^", linewidth=2, label="99th Percentile Latency (ms)", color="#9467bd")

        ax.set_xlabel("Workload Event Volume", fontsize=11, fontweight="bold")
        ax.set_ylabel("Latency (Milliseconds)", fontsize=11, fontweight="bold")
        ax.set_title("End-to-End Processing Latency vs Workload Volume", fontsize=12, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plot2_path = self.plots_dir / "latency_comparison.png"
        plt.savefig(plot2_path, dpi=300)
        plt.close()
        print(f"  Saved {plot2_path}")

        # Plot 3: Detector Quality Comparison (Precision, Recall, F1, FPR)
        fig, ax = plt.subplots(figsize=(9, 5))
        det_names = [m["detector"] for m in offline_metrics]
        precisions = [m["precision"] for m in offline_metrics]
        recalls = [m["recall"] for m in offline_metrics]
        f1_scores = [m["f1_score"] for m in offline_metrics]
        fprs = [m["false_positive_rate"] for m in offline_metrics]

        x = np.arange(len(det_names))
        w = 0.2
        ax.bar(x - 1.5*w, precisions, w, label="Precision", color="#3498db")
        ax.bar(x - 0.5*w, recalls, w, label="Recall", color="#2ecc71")
        ax.bar(x + 0.5*w, f1_scores, w, label="F1-Score", color="#e67e22")
        ax.bar(x + 1.5*w, fprs, w, label="FPR (False Positive Rate)", color="#e74c3c")

        ax.set_xlabel("Anomaly Detection Method", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score / Rate [0.0 - 1.0]", fontsize=11, fontweight="bold")
        ax.set_title("Comparative Evaluation of Anomaly Detectors on AI4I Holdout Test Set", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(det_names, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.tight_layout()
        plot3_path = self.plots_dir / "detector_metrics.png"
        plt.savefig(plot3_path, dpi=300)
        plt.close()
        print(f"  Saved {plot3_path}")

        # Plot 4: Baseline vs Extension Workload Throughput
        fig, ax = plt.subplots(figsize=(8, 5))
        c_events = [r["workload_events"] for r in comparison_results]
        base_tp = [r["baseline_throughput_rows_per_sec"] for r in comparison_results]
        ext_tp = [r["extension_throughput_rows_per_sec"] for r in comparison_results]

        x = np.arange(len(c_events))
        width = 0.35
        ax.bar(x - width/2, base_tp, width, label="Baseline (Simple State Temp Aggregations)", color="#7f7f7f")
        ax.bar(x + width/2, ext_tp, width, label="Extension (Multivariate Anomaly Analytics)", color="#17becf")

        ax.set_xlabel("Workload Event Volume", fontsize=11, fontweight="bold")
        ax.set_ylabel("Throughput (Rows/sec)", fontsize=11, fontweight="bold")
        ax.set_title("Throughput Comparison: Baseline vs Extended Multivariate Pipeline", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{e:,}" for e in c_events])
        ax.legend()
        plt.tight_layout()
        plot4_path = self.plots_dir / "baseline_vs_extension.png"
        plt.savefig(plot4_path, dpi=300)
        plt.close()
        print(f"  Saved {plot4_path}")


def main():
    parser = argparse.ArgumentParser(description="Big Data Experiment and Benchmark Suite")
    parser.add_argument("--servers", default=KAFKA_BOOTSTRAP_SERVERS, help="Kafka bootstrap servers")
    parser.add_argument("--scales", nargs="+", type=int, default=[10000, 50000, 100000], help="Workload event counts")
    args = parser.parse_args()

    spark = get_spark_session(app_name="BenchmarkSuite")
    spark.sparkContext.setLogLevel("WARN")

    runner = BenchmarkRunner(bootstrap_servers=args.servers)

    try:
        # Run Workload Scaling
        scaling_results = runner.run_scaling_benchmark(event_counts=args.scales, spark=spark)

        # Run Baseline Comparison
        comparison_results = runner.run_baseline_comparison(spark=spark, event_counts=[10000, 50000])

        # Load offline metrics
        offline_metrics_path = OUTPUT_DIR / "offline_test_metrics.json"
        if offline_metrics_path.exists():
            with open(offline_metrics_path, "r", encoding="utf-8") as f:
                offline_metrics = json.load(f)
        else:
            offline_metrics = []

        # Generate figures
        runner.generate_benchmark_plots(scaling_results, comparison_results, offline_metrics)

        # Save machine-readable output
        benchmark_bundle = {
            "environment": runner.env_info,
            "scaling_benchmark": scaling_results,
            "baseline_comparison": comparison_results,
            "offline_detector_metrics": offline_metrics,
            "benchmark_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        json_out = BENCHMARK_OUTPUT_DIR / "benchmark_results.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(benchmark_bundle, f, indent=2)

        # Save summary CSV
        csv_out = BENCHMARK_OUTPUT_DIR / "benchmark_summary.csv"
        pd.DataFrame(scaling_results).to_csv(csv_out, index=False)

        print("\n" + "=" * 70)
        print("BENCHMARK SUITE COMPLETED SUCCESSFULLY")
        print(f"  Results saved to: {json_out}")
        print(f"  Summary saved to: {csv_out}")
        print(f"  Plots saved to:   {BENCHMARK_OUTPUT_DIR / 'plots'}")
        print("=" * 70)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
