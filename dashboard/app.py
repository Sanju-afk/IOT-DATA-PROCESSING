"""
Industrial IoT Anomaly Detection & Scalable Stream Analytics Dashboard.
Built with Streamlit.
Visualizes real-time Kafka-Spark stream telemetry, offline detector evaluation,
sensor feature distributions, failure modes, and big data benchmark scalability.
Supports live streaming mode and offline fallback.
"""

import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="Industrial IoT Stream Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
BENCHMARK_DIR = OUTPUT_DIR / "benchmarks"
PLOTS_DIR = BENCHMARK_DIR / "plots"
STREAM_RESULTS_CSV = OUTPUT_DIR / "latest_stream_predictions.csv"
STREAM_METRICS_LOG = OUTPUT_DIR / "stream_metrics_log.jsonl"
OFFLINE_PRED_CSV = OUTPUT_DIR / "offline_test_predictions.csv"
OFFLINE_METRICS_JSON = OUTPUT_DIR / "offline_test_metrics.json"
BENCHMARK_RESULTS_JSON = BENCHMARK_DIR / "benchmark_results.json"
BENCHMARK_SUMMARY_CSV = BENCHMARK_DIR / "benchmark_summary.csv"


# The stream rewrites STREAM_RESULTS_CSV every micro-batch; older than this, it's a finished run.
LIVE_STALE_SEC = 15


@st.cache_data(max_entries=4)
def read_csv_version(path: str, mtime: float) -> pd.DataFrame:
    """Reads a CSV once per file version (mtime is part of the cache key)."""
    return pd.read_csv(path)


def load_stream_or_fallback_data():
    """
    Returns (df, source, age_sec). source is "live" when the streaming job wrote
    within LIVE_STALE_SEC, "stream" for an earlier run's output, else "offline"
    for the holdout predictions.
    """
    if STREAM_RESULTS_CSV.exists():
        mtime = STREAM_RESULTS_CSV.stat().st_mtime
        df = read_csv_version(str(STREAM_RESULTS_CSV), mtime)
        if not df.empty:
            age = time.time() - mtime
            return df, ("live" if age <= LIVE_STALE_SEC else "stream"), age

    if OFFLINE_PRED_CSV.exists():
        return read_csv_version(str(OFFLINE_PRED_CSV), OFFLINE_PRED_CSV.stat().st_mtime), "offline", None

    return pd.DataFrame(), "offline", None


def load_recent_batch_metrics(n: int = 40) -> pd.DataFrame:
    """Last n micro-batch records from the streaming job's metrics log."""
    if not STREAM_METRICS_LOG.exists():
        return pd.DataFrame()
    with open(STREAM_METRICS_LOG, "r", encoding="utf-8") as f:
        lines = f.readlines()[-n:]
    df = pd.DataFrame([json.loads(line) for line in lines if line.strip()])
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@st.cache_data
def load_metrics_and_benchmarks():
    """Loads offline metrics and benchmark records."""
    metrics = []
    if OFFLINE_METRICS_JSON.exists():
        with open(OFFLINE_METRICS_JSON, "r") as f:
            metrics = json.load(f)

    benchmarks = {}
    if BENCHMARK_RESULTS_JSON.exists():
        with open(BENCHMARK_RESULTS_JSON, "r") as f:
            benchmarks = json.load(f)

    return metrics, benchmarks


def render_status_and_kpis():
    data_df, source, age = load_stream_or_fallback_data()

    if source == "live":
        st.success(f"LIVE: streaming job last wrote a micro-batch {age:.0f}s ago")
    elif source == "stream":
        st.info(f"Stream idle: showing the last streaming run (updated {age / 60:.0f} min ago)")
    else:
        st.info("Offline evaluation mode: showing holdout test-split predictions")

    if data_df.empty:
        return
    n_total = len(data_df)

    def count(col):
        return int(data_df[col].sum()) if col in data_df.columns else 0

    n_failures = count("machine_failure")
    cols = st.columns(6)
    cols[0].metric("Events (recent window)", f"{n_total:,}")
    cols[1].metric("True Failures", f"{n_failures:,}", f"{(n_failures/max(n_total,1)):.2%} rate")
    for col, (label, pred) in zip(cols[2:], [
        ("Threshold Anomalies", "pred_threshold"),
        ("Z-Score Anomalies", "pred_zscore"),
        ("Isolation Forest", "pred_iforest"),
        ("Ensemble Anomalies", "pred_ensemble"),
    ]):
        n = count(pred)
        col.metric(label, f"{n:,}", f"{(n/max(n_total,1)):.2%}")


def render_live_telemetry_tab():
    data_df, source, _ = load_stream_or_fallback_data()
    st.subheader("Industrial Milling Machine Sensor Telemetry")
    if data_df.empty:
        st.warning("No telemetry records loaded. Start the producer and streaming pipeline to view live stream data.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**Sensor Timeseries Window (Recent Events)**")
        features_to_plot = ["torque", "rotational_speed", "air_temperature", "process_temperature"]
        available_feats = [f for f in features_to_plot if f in data_df.columns]
        sample_df = data_df.tail(200).copy()
        sample_df["event_index"] = np.arange(len(sample_df))

        selected_feature = st.selectbox(
            "Select sensor feature to visualize:", available_feats, index=0, key="live_feature"
        )
        st.line_chart(sample_df.set_index("event_index")[[selected_feature]])

    with c2:
        st.markdown("**Product Variant Distribution**")
        if "product_type" in data_df.columns:
            st.bar_chart(data_df["product_type"].value_counts())

    if source != "offline":
        batches = load_recent_batch_metrics()
        if not batches.empty:
            st.markdown("### Micro-Batch Activity (Spark Structured Streaming)")
            b1, b2 = st.columns(2)
            with b1:
                st.markdown("**Ensemble anomalies per micro-batch**")
                st.bar_chart(batches.set_index("timestamp")[["anomalies_ensemble"]])
            with b2:
                st.markdown("**Mean event-to-scoring latency (ms)**")
                st.line_chart(batches.set_index("timestamp")[["mean_latency_ms"]])

    st.markdown("---")
    st.markdown("### Failure Modes & Anomaly Breakdown")
    f_cols = ["twf", "hdf", "pwf", "osf", "rnf"]
    f_present = [f for f in f_cols if f in data_df.columns]
    if f_present:
        f_sums = data_df[f_present].sum()
        f_df = pd.DataFrame({
            "Failure Mode": ["Tool Wear (TWF)", "Heat Dissipation (HDF)", "Power Failure (PWF)", "Overstrain (OSF)", "Random (RNF)"],
            "Trigger Count": [f_sums.get(k, 0) for k in f_cols]
        })
        st.dataframe(f_df, hide_index=True, use_container_width=True)

    st.markdown("### Recent Scored Telemetry Records")
    cols_preview = [
        "udi", "product_id", "product_type", "air_temperature", "process_temperature",
        "rotational_speed", "torque", "tool_wear", "machine_failure",
        "pred_threshold", "pred_zscore", "pred_iforest", "pred_ensemble"
    ]
    cols_avail = [c for c in cols_preview if c in data_df.columns]
    st.dataframe(data_df[cols_avail].tail(25).iloc[::-1], use_container_width=True)


def main():
    st.title("Real-Time Industrial IoT Anomaly Detection & Stream Analytics")
    st.markdown(
        "**Scalable Distributed Telemetry Pipeline** using Apache Kafka, Apache Spark Structured Streaming, "
        "Domain Physics, Z-Score Distance, and Multivariate Isolation Forest."
    )

    metrics_list, benchmark_bundle = load_metrics_and_benchmarks()

    # Sidebar Controls
    st.sidebar.header("Stream & Pipeline Mode")
    auto_refresh = st.sidebar.toggle("Auto-refresh live stream (every 2s)", value=True)
    run_every = "2s" if auto_refresh else None

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Reference")
    st.sidebar.markdown(
        "**UCI AI4I 2020 Predictive Maintenance**\n"
        "- 10,000 milling machine records\n"
        "- Features: Temperatures, Speed, Torque, Tool Wear\n"
        "- Ground truth: 3.39% machine failure rate\n"
        "- Reference: *S. Matzka (2020), IEEE AI4I*"
    )

    # Only these fragments rerun on the refresh timer; the static tabs below render once.
    st.fragment(render_status_and_kpis, run_every=run_every)()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Live Telemetry & Sensor Trends",
        "Anomaly Detection Performance",
        "Big Data Benchmarks & Scalability",
        "System Architecture & Methods"
    ])

    # TAB 1: Telemetry & Trends
    with tab1:
        st.fragment(render_live_telemetry_tab, run_every=run_every)()

    # TAB 2: Performance Evaluation
    with tab2:
        st.subheader("Comparative Evaluation Across Anomaly Detection Methods")
        st.markdown(
            "Detectors are evaluated on the **2,000-sample holdout test split** (strictly unseen during training). "
            "Because machine failures comprise only **3.4%** of the dataset, accuracy is misleading (a naive classifier predicting 0 achieves 96.6% accuracy but 0% recall). "
            "We focus on **Precision, Recall, F1-Score, False Positive Rate (FPR), ROC-AUC, and PR-AUC**."
        )

        if metrics_list:
            m_df = pd.DataFrame(metrics_list)
            st.dataframe(m_df, hide_index=True, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Precision, Recall & F1 Comparison**")
                chart_df = m_df.set_index("detector")[["precision", "recall", "f1_score"]]
                st.bar_chart(chart_df)
            with c2:
                st.markdown("**False Positive Rate (FPR) vs Specificity**")
                fpr_df = m_df.set_index("detector")[["false_positive_rate", "specificity"]]
                st.bar_chart(fpr_df)

            st.markdown("---")
            st.markdown("### Key Methodological Insights")
            st.markdown(
                """
                - **Method A (Domain Thresholds):** Achieves **97.06% Recall** (catches almost all machine failures) by modeling the explicit physical equations (power bounds, tool wear limits, heat dissipation temp delta). However, operating near physical boundaries also triggers **150 False Alarms (Precision = 30.56%)**.
                - **Method B (Standardized Z-Score):** Has very low false alarms (**FPR = 1.24%**), but suffers low recall (**17.65%**). Because multivariate failures (such as high torque combined with high speed) occur within individual 3-sigma univariate envelopes, univariate distance metrics fail to capture nonlinear correlations.
                - **Method C (Multivariate Isolation Forest):** Isolates multivariate interactions in tree partitions with balanced precision (**28.74%**) and recall (**36.76%**) with low FPR (**3.21%**).
                - **Majority Vote (>=2 of 3, hard voting):** Demanding binary consensus across detectors improves precision to **47.37%** but collapses recall to **39.71%** — it discards each detector's confidence and treats a borderline signal the same as a decisive one, so a strong-but-lone physics signal gets outvoted.
                - **OR / Any-detector (safety-net voting):** Flags an event if *any* detector fires. Matches the Threshold detector's **97.06% recall** but at the cost of precision (**24.72%**) and the highest false-alarm rate (**10.40% FPR**) — a reasonable early-warning mode, not a final alarm.
                - **Method D (Stacked Ensemble — Logistic Regression meta-model):** Instead of voting on binary outputs, a cost-sensitive Logistic Regression learns how to weigh the three detectors' *continuous* scores. Its decision threshold is tuned to maximize F2 (recall-weighted F-score) using 5-fold cross-validated out-of-fold predictions across the full training split — exposing the tuning to all 271 training-split failures instead of a single small validation slice. Result: **91.18% Recall** and **51.24% Precision** (F1 = 0.6561, F2 = 0.7888, FPR = 3.05%, ROC-AUC = 0.9771) — missing only 6 of 68 failures versus 41 missed by hard majority voting, by preserving detector confidence instead of discarding it.
                """
            )
        else:
            st.info("Run `python -m src.models.train_offline` to generate offline holdout metrics.")

    # TAB 3: Benchmarks & Scalability
    with tab3:
        st.subheader("Big Data Scalability: Throughput & Latency Benchmarks")
        st.markdown(
            "Empirical benchmark results run on WSL with Apache Kafka 4.1.2 KRaft and Apache Spark 4.2.0 Structured Streaming."
        )

        if BENCHMARK_SUMMARY_CSV.exists():
            summary_df = pd.read_csv(BENCHMARK_SUMMARY_CSV)
            st.markdown("### Scaling Summary (10k, 50k, 100k Events)")
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if (PLOTS_DIR / "throughput_scaling.png").exists():
                st.image(str(PLOTS_DIR / "throughput_scaling.png"), caption="Pipeline Throughput vs Event Volume", use_container_width=True)
            if (PLOTS_DIR / "detector_metrics.png").exists():
                st.image(str(PLOTS_DIR / "detector_metrics.png"), caption="Detector Accuracy & Error Rates", use_container_width=True)

        with col_p2:
            if (PLOTS_DIR / "latency_comparison.png").exists():
                st.image(str(PLOTS_DIR / "latency_comparison.png"), caption="End-to-End Latency Across Workload Scales", use_container_width=True)

        if benchmark_bundle.get("environment"):
            st.markdown("### Hardware & Software Test Environment")
            env = benchmark_bundle["environment"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("OS & Kernel", f"{env.get('os')} ({env.get('platform_release', '')[:14]})")
            c2.metric("CPU Cores", f"{env.get('cpu_count_logical')} logical")
            c3.metric("System RAM", f"{env.get('total_ram_gb')} GB")
            c4.metric("Python Version", f"{env.get('python_version')}")

    # TAB 4: Architecture & Documentation
    with tab4:
        st.subheader("Pipeline Architecture & Design Choices")
        st.markdown(
            """
            ```mermaid
            flowchart LR
                A["AI4I 2020 CSV / Synthetic Engine"] -->|JSON Events| B["Python Kafka Producer"]
                B -->|iot_telemetry Topic| C["Apache Kafka 4.1.2 (KRaft)"]
                C -->|Structured Streaming| D["Apache Spark 4.2.0"]
                D --> E["Schema Validation & Physics Features"]
                E --> F["Tumbling Window Aggregations (10s)"]
                E --> G["Modular Anomaly Detectors"]
                G --> H["Method A: Domain Thresholds"]
                G --> I["Method B: Z-Score Distance"]
                G --> J["Method C: Isolation Forest"]
                G --> K["Method D: Stacked Ensemble (Logistic Regression)"]
                K -->|iot_anomalies Topic| L["Kafka Anomaly Topic"]
                K -->|Hive Table, Parquet by event_date| W["Hive Warehouse: iot_analytics.telemetry_scored"]
                K -->|CSV Sinks| M["Durable Storage (output/)"]
                M --> N["Streamlit Live Dashboard"]
            ```
            """
        )
        st.markdown(
            """
            ### Design Choices
            | Component | Implementation | Rationale |
            |---|---|---|
            | **Execution Runtime** | Python 3.14, Spark 4.2.0 | Current, supported runtime versions |
            | **Ingestion Engine** | Spark Structured Streaming Kafka Source | Modern fault-tolerant streaming architecture |
            | **Data Schema** | 14-feature multivariate milling machine telemetry | Realistic industrial IoT sensor dynamics |
            | **Physics Modeling** | Power equations, tool wear wearout, heat dissipation | Direct alignment with AI4I milling machine physics |
            | **Analytics** | Domain Thresholds, Z-Scores, Isolation Forest, Ensemble | Anomaly detection research question |
            | **Persistence & Sinks** | Hive table (Parquet, partitioned by event date), CSV micro-batch files, Kafka anomaly topic | SQL-queryable warehouse for downstream analytics; topic for real-time alerting |
            | **Visualization** | Interactive Streamlit Dashboard | Real-time monitoring and empirical reporting |
            """
        )


if __name__ == "__main__":
    main()
