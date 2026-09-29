"""
Industrial IoT Anomaly Detection & Scalable Stream Analytics Dashboard.
Built with Streamlit.
Visualizes real-time Kafka-Spark stream telemetry, offline detector evaluation,
sensor feature distributions, failure modes, and big data benchmark scalability.
Supports live streaming mode and offline fallback.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="Industrial IoT Stream Analytics",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
BENCHMARK_DIR = OUTPUT_DIR / "benchmarks"
PLOTS_DIR = BENCHMARK_DIR / "plots"
STREAM_RESULTS_CSV = OUTPUT_DIR / "latest_stream_predictions.csv"
OFFLINE_PRED_CSV = OUTPUT_DIR / "offline_test_predictions.csv"
OFFLINE_METRICS_JSON = OUTPUT_DIR / "offline_test_metrics.json"
BENCHMARK_RESULTS_JSON = BENCHMARK_DIR / "benchmark_results.json"
BENCHMARK_SUMMARY_CSV = BENCHMARK_DIR / "benchmark_summary.csv"


@st.cache_data(ttl=5)
def load_stream_or_fallback_data():
    """Loads stream telemetry if available, else falls back to offline holdout results."""
    is_live = False
    if STREAM_RESULTS_CSV.exists() and STREAM_RESULTS_CSV.stat().st_size > 100:
        try:
            df = pd.read_csv(STREAM_RESULTS_CSV)
            if len(df) > 0:
                is_live = True
                return df, is_live
        except Exception:
            pass

    if OFFLINE_PRED_CSV.exists():
        df = pd.read_csv(OFFLINE_PRED_CSV)
        return df, False

    return pd.DataFrame(), False


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


def main():
    st.title("⚙️ Real-Time Industrial IoT Anomaly Detection & Stream Analytics")
    st.markdown(
        "**Scalable Distributed Telemetry Pipeline** using Apache Kafka, Apache Spark Structured Streaming, "
        "Domain Physics, Z-Score Distance, and Multivariate Isolation Forest."
    )

    data_df, is_live = load_stream_or_fallback_data()
    metrics_list, benchmark_bundle = load_metrics_and_benchmarks()

    # Sidebar Controls
    st.sidebar.header("🕹️ Stream & Pipeline Mode")
    if is_live:
        st.sidebar.success("🟢 Active Live Stream Telemetry Detected")
    else:
        st.sidebar.info("🔵 Offline Evaluation Mode (Fallback Loaded)")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Reference")
    st.sidebar.markdown(
        "**UCI AI4I 2020 Predictive Maintenance**\n"
        "- 10,000 milling machine records\n"
        "- Features: Temperatures, Speed, Torque, Tool Wear\n"
        "- Ground truth: 3.39% machine failure rate\n"
        "- Reference: *S. Matzka (2020), IEEE AI4I*"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏛️ Starting Baseline Repo")
    st.sidebar.markdown(
        "[Yugo Kato: Spark & Kafka IoT Analytics](https://github.com/yugokato/Spark-and-Kafka_IoT-Data-Processing-and-Analytics)\n"
        "- Legacy Spark 2.0 DStream / Python 2\n"
        "- Single-sensor state aggregations\n"
        "- Extended with modern Spark Structured Streaming, multivariate ML, and automated benchmarking."
    )

    # Top KPI Metrics Row
    if not data_df.empty:
        n_total = len(data_df)
        n_failures = int(data_df["machine_failure"].sum()) if "machine_failure" in data_df.columns else 0
        n_thresh = int(data_df["pred_threshold"].sum()) if "pred_threshold" in data_df.columns else 0
        n_zscore = int(data_df["pred_zscore"].sum()) if "pred_zscore" in data_df.columns else 0
        n_iforest = int(data_df["pred_iforest"].sum()) if "pred_iforest" in data_df.columns else 0
        n_ensemble = int(data_df["pred_ensemble"].sum()) if "pred_ensemble" in data_df.columns else 0

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Events Ingested", f"{n_total:,}")
        col2.metric("True Failures", f"{n_failures:,}", f"{(n_failures/max(n_total,1)):.2%} rate")
        col3.metric("Threshold Anomalies", f"{n_thresh:,}", f"{(n_thresh/max(n_total,1)):.2%}")
        col4.metric("Z-Score Anomalies", f"{n_zscore:,}", f"{(n_zscore/max(n_total,1)):.2%}")
        col5.metric("Isolation Forest", f"{n_iforest:,}", f"{(n_iforest/max(n_total,1)):.2%}")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Live Telemetry & Sensor Trends",
        "🎯 Anomaly Detection Performance",
        "⚡ Big Data Benchmarks & Scalability",
        "📐 System Architecture & Methods"
    ])

    # TAB 1: Telemetry & Trends
    with tab1:
        st.subheader("Industrial Milling Machine Sensor Telemetry")
        if not data_df.empty:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("**Sensor Timeseries Window (Recent Events)**")
                features_to_plot = ["torque", "rotational_speed", "air_temperature", "process_temperature"]
                available_feats = [f for f in features_to_plot if f in data_df.columns]
                sample_df = data_df.tail(200).copy()
                sample_df["event_index"] = np.arange(len(sample_df))

                selected_feature = st.selectbox("Select sensor feature to visualize:", available_feats, index=0)
                st.line_chart(sample_df.set_index("event_index")[[selected_feature]])

            with c2:
                st.markdown("**Product Variant Distribution**")
                if "product_type" in data_df.columns:
                    type_counts = data_df["product_type"].value_counts()
                    st.bar_chart(type_counts)

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
            st.dataframe(data_df[cols_avail].tail(25), use_container_width=True)
        else:
            st.warning("No telemetry records loaded. Start the producer and streaming pipeline to view live stream data.")

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
                - **Method D (Ensemble Majority Vote):** Demanding consensus across detectors achieves the **highest Precision (47.37%)** and lowest False Positive Rate (**1.55%**), demonstrating the power of multi-method validation in industrial operations.
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
            if (PLOTS_DIR / "baseline_vs_extension.png").exists():
                st.image(str(PLOTS_DIR / "baseline_vs_extension.png"), caption="Baseline (State Aggs) vs Proposed Extension (ML Detection)", use_container_width=True)

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
        st.subheader("Pipeline Architecture & Modifications from Baseline")
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
                G --> K["Method D: Ensemble Vote"]
                K -->|iot_anomalies Topic| L["Kafka Anomaly Topic"]
                K -->|Parquet / CSV Sinks| M["Durable Storage (output/)"]
                M --> N["Streamlit Live Dashboard"]
            ```
            """
        )
        st.markdown(
            """
            ### Summary of Baseline Modifications
            | Component | Original Baseline (Kato) | Proposed Extension | Rationale |
            |---|---|---|---|
            | **Execution Runtime** | Python 2.7, Spark 1.x/2.0-preview | Python 3.14, Spark 4.2.0 | Deprecation of Python 2 and Spark DStreams |
            | **Ingestion Engine** | Legacy `KafkaUtils.createDirectStream` | Spark Structured Streaming Kafka Source | Modern fault-tolerant streaming architecture |
            | **Data Schema** | Single synthetic temperature float | 14-feature multivariate milling machine telemetry | Realistic industrial IoT sensor dynamics |
            | **Physics Modeling** | Random walk per US state | Power equations, tool wear wearout, heat dissipation | Direct alignment with AI4I milling machine physics |
            | **Analytics** | Average temperature, sensor counts | Domain Thresholds, Z-Scores, Isolation Forest, Ensemble | Anomaly detection research question |
            | **Persistence & Sinks** | Console `pprint()` only | Parquet, CSV micro-batch sinks, Kafka anomaly topic | Durable storage for downstream alerting and analytics |
            | **Visualization** | None | Interactive Streamlit Dashboard | Real-time monitoring and empirical reporting |
            """
        )


if __name__ == "__main__":
    main()
