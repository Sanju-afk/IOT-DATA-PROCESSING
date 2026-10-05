# Real-Time Industrial IoT Anomaly Detection and Scalable Stream Analytics Using Apache Kafka and Apache Spark

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![Apache Spark 4.2.0](https://img.shields.io/badge/Apache%20Spark-4.2.0-orange.svg)](https://spark.apache.org/)
[![Apache Kafka 4.1.2](https://img.shields.io/badge/Apache%20Kafka-4.1.2%20(KRaft)-black.svg)](https://kafka.apache.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Problem Statement & Motivation

Industrial machinery (milling machines, computer numerical control (CNC) centers, turbines, and pumps) operates under continuous, severe mechanical and thermodynamic stresses. Telemetry streams emitted from such machines encompass multivariate parameters—ambient air and internal process temperatures, spindle rotational speeds, shaft torques, and cumulative tool wear. 

In traditional operational settings, two primary architectural limitations impede timely maintenance:
1. **Batch Processing Delays:** Telemetry collected over hours or shifts is analyzed retrospectively, detecting degradation only *after* tool fracture or workpiece damage has occurred.
2. **Univariate Static Thresholds:** Alarm triggers based solely on single-sensor extremes frequently fail to detect complex failure modes that emerge from multi-variable interactions (such as moderate torque occurring at low spindle speeds causing inadequate heat dissipation), or generate excessive false positive alarms that desensitize factory operators.

This project investigates how a modern distributed stream processing pipeline can ingest, validate, and analyze multivariate industrial IoT telemetry in real time, comparing physical domain rules, statistical distance baselines, and multivariate machine learning models.

---

## 2. Research Question

> **Research Question:** *How do multivariate anomaly detection methods (Isolation Forest) compare with simple threshold-based detection in detection quality (Precision, Recall, F1, FPR), and how does the Kafka–Spark pipeline behave as input event volume and stream throughput scale up to 100,000 events?*

---

## 3. Attribution of Baseline Work

This project builds upon and modernizes the open-source starting point:
- **Original Repository:** [yugokato/Spark-and-Kafka_IoT-Data-Processing-and-Analytics](https://github.com/yugokato/Spark-and-Kafka_IoT-Data-Processing-and-Analytics)
- **Original Author:** Yugo Kato
- **Baseline Architecture:** Ingested simulated single-variable ambient temperature data mapped across 50 US states using Python 2.7 and legacy Spark 1.x/2.0-preview DStreams (`KafkaUtils.createDirectStream`).
- **Baseline Analytics:** Computed state-level average temperatures, total message counts, and distinct sensor counts via console `pprint()`.
- **Modifications in This Work:** We preserved the baseline implementation in [`original_repo/`](original_repo/) and developed a modernized, production-grade streaming pipeline featuring Spark Structured Streaming, Kafka KRaft, physics-based feature engineering, multi-detector anomaly analytics, automated benchmarking, and an interactive Streamlit dashboard.

---

## 4. Dataset Source & Academic Citations

The project uses the public **UCI AI4I 2020 Predictive Maintenance Dataset** as its empirical foundation:
- **UCI Repository URL:** [https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Dataset Properties:** 10,000 synthetic industrial records reflecting realistic milling machine telemetry, containing 14 attributes and ground truth failure indicators (3.39% empirical machine failure prevalence).

### Academic Citations
1. **Dataset Origin & Physics Formulation:**
   > Matzka, S. (2020). *"Explainable Artificial Intelligence for Predictive Maintenance Applications,"* 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), pp. 69–74. [doi:10.1109/AI4I49448.2020.00023](https://doi.org/10.1109/AI4I49448.2020.00023).
2. **Contextual Predictive Maintenance Survey:**
   > Carvalho, M. S., et al. (2019). *"A systematic literature review of machine learning in the context of predictive maintenance,"* Computers & Industrial Engineering, 137, 106024. [doi:10.1016/j.cie.2019.106024](https://doi.org/10.1016/j.cie.2019.106024).
3. **Repository Reference:**
   > Dua, D. and Graff, C. (2019). *UCI Machine Learning Repository* [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

---

## 5. System Architecture

The pipeline implements an end-to-end distributed stream processing topology:

```
[ AI4I 2020 Dataset / Synthetic Stream Generator ]
                       |
                       v
         [ Python Kafka Telemetry Producer ]
                       |
                       v (Topic: iot_telemetry, Port 9092)
       [ Apache Kafka 4.1.2 Cluster (KRaft Mode) ]
                       |
                       v (Structured Streaming Ingestion)
     [ Apache Spark 4.2.0 Engine (Micro-batch: 2s) ]
        |---> Schema Parsing & Validation
        |---> Physics Feature Engineering (Power, Temp Diff, Overstrain)
        |---> Tumbling Window Aggregations (10-second Windows)
        |---> Modular Multi-Detector Scoring:
                 ├── Method A: Physics Domain Thresholds
                 ├── Method B: Standardized Z-Score Distance
                 ├── Method C: Multivariate Isolation Forest
                 └── Method D: Stacked Ensemble (Logistic Regression meta-model)
        |
        +---> Emits Anomalies to Kafka (Topic: iot_anomalies)
        +---> Writes Enriched Telemetry to Parquet/CSV (output/stream_results/)
        +---> Emits Windowed Aggregates to Kafka (Topic: iot_aggregates)
                       |
                       v
       [ Interactive Streamlit Dashboard (Port 8501) ]
```

---

## 6. Physics Equations & Data Schema

The AI4I milling machine dataset models five distinct physical failure mechanisms:

1. **Heat Dissipation Failure (HDF):** Inadequate convective heat removal occurs when the difference between process temperature and ambient air temperature is under $8.6\text{ K}$ while rotational speed is below $1380\text{ rpm}$:
   $$\Delta T = T_{\text{process}} - T_{\text{air}} < 8.6\text{ K}\quad \land \quad \text{Speed} \le 1380\text{ rpm}$$
2. **Power Failure (PWF):** The product of torque and angular velocity represents instantaneous mechanical power:
   $$P = \tau \cdot \left(\text{speed}_{\text{rpm}} \cdot \frac{2\pi}{60}\right)\quad [\text{Watts}]$$
   A power failure occurs if $P < 3500\text{ W}$ (stall/underload) or $P > 9000\text{ W}$ (overload).
3. **Tool Wear Failure (TWF):** Progressive cutting tool abrasive wear reaches replacement wearout threshold between $200$ and $240$ minutes ($W_{\text{tool}} \ge 200\text{ min}$).
4. **Overstrain Failure (OSF):** The product of accumulated tool wear and cutting torque exceeds structural shearing limits based on product quality variant:
   $$S = W_{\text{tool}} \cdot \tau > \begin{cases} 11,000\text{ min}\cdot\text{Nm} & (\text{Variant L}) \\ 12,000\text{ min}\cdot\text{Nm} & (\text{Variant M}) \\ 13,000\text{ min}\cdot\text{Nm} & (\text{Variant H}) \end{cases}$$
5. **Random Failure (RNF):** Independent stochastic breakdown with a constant $0.1\%$ occurrence probability.

---

## 7. Comparative Experimental Results

Threshold, Z-Score, and Isolation Forest are fit on the full $80\%$ training split ($8,000$ samples; Z-Score/Isolation Forest use only its $7,729$ normal rows). The ensemble meta-model is fit on $5$-fold stratified cross-validated out-of-fold (OOF) scores over that entire training split — exposing its Logistic Regression fit and F2 threshold sweep to all $271$ training-split failures, rather than only the failures that happen to land in one held-out slice. The final $20\%$ holdout test split ($2,000$ samples, $68$ ground-truth machine failures, $3.4\%$ class prevalence) is unseen by every component and used only for the table below.

### 7.1 Detection Quality Metrics

| Detector Method | Precision | Recall | F1-Score | F2-Score | FPR | ROC-AUC | PR-AUC | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| **Method A: Domain Thresholds** | 30.56% | **97.06%** | 0.4648 | 0.6762 | 7.76% | 0.9790 | **0.8799** | 66 | 150 | 2 |
| **Method B: Standardized Z-Score** | 33.33% | 17.65% | 0.2308 | 0.1948 | **1.24%** | 0.8666 | 0.2395 | 12 | 24 | 56 |
| **Method C: Isolation Forest** | 28.74% | 36.76% | 0.3226 | 0.3482 | 3.21% | 0.8926 | 0.2451 | 25 | 62 | 43 |
| Hard majority vote ($\ge 2$ of 3) | 47.37% | 39.71% | 0.4320 | 0.4103 | 1.55% | N/A | N/A | 27 | 30 | 41 |
| OR / any-detector vote | 24.72% | 97.06% | 0.3940 | 0.6122 | 10.40% | N/A | N/A | 66 | 201 | 2 |
| **Method D: Stacked Ensemble** | **51.24%** | **91.18%** | **0.6561** | **0.7888** | 3.05% | 0.9771 | 0.7241 | 62 | 59 | 6 |

#### Methodological Takeaways:
- **The Class Imbalance Problem:** In a dataset with 3.4% failure prevalence, a naive baseline predicting all normal samples achieves **96.6% accuracy but 0.0% recall**. Precision, Recall, F1/F2, and False Positive Rate (FPR) are the only valid operational evaluation metrics.
- **Univariate vs Multivariate Trade-off:** Standardized Z-scores achieve the lowest false positive rate (1.24%) but miss over 82% of failures because correlated multivariate anomalies remain within standard 3-sigma single-feature envelopes.
- **Physical Rules:** Physical threshold boundaries catch 97.1% of failures (highest recall among single detectors), but suffer 150 false alarms because operating near stress boundaries is not always fatal.
- **Hard voting discards confidence:** Requiring 2-of-3 binary votes to agree *improves* precision over any single detector (47.37%) but *collapses* recall to 39.71% — a strong-but-lone physics signal gets outvoted by two detectors that were only weakly uncertain, throwing away useful evidence.
- **Stacked ensemble wins on recall and F2 while still beating hard voting on precision:** Rather than voting on binary outputs, a cost-sensitive Logistic Regression meta-model (`UnifiedPipelineDetector.fit_meta_from_scores`, see [`src/models/detectors.py`](src/models/detectors.py) and `build_oof_meta_features` in [`src/models/train_offline.py`](src/models/train_offline.py)) learns how to weigh the three detectors' *continuous* scores, with its decision threshold swept over cross-validated OOF predictions to maximize F2 (recall-weighted F-score) instead of assuming P≥0.5. Versus hard majority voting it recovers **+51.5 points recall and +3.9 points precision** (missing only 6 of 68 failures instead of 41), at the cost of ~29 more false alarms out of 2,000 test events — a trade this project accepts because a missed machine failure is far costlier than a false alarm.

---

### 7.2 Scalability & Ingestion Throughput Benchmarks

Empirical scalability benchmarks conducted on an 8-core WSL 2 environment:

| Workload ($N$) | Producer Throughput | Processing Throughput | Duration | 95th % Latency | Memory Footprint |
|---|---|---|---|---|---|
| **10,000 events** | 4,037 evt/sec | 4,631 rows/sec | 2.16 sec | 89.7 sec* | 342.8 MB |
| **50,000 events** | 4,429 evt/sec | **6,509 rows/sec** | 7.68 sec | 454.5 sec* | 425.8 MB |
| **100,000 events** | 4,264 evt/sec | 6,286 rows/sec | 15.91 sec | 907.6 sec* | 524.5 MB |

*(Note on Latency:* Latency measures total wall-clock lag from event generation to end of unthrottled burst processing).*

---

### 7.3 Baseline vs Extension Performance Comparison

Comparing the original repository's state temperature aggregations against our multivariate anomaly detection workload:

| Event Count | Baseline Workload (State Aggregations) | Extension Workload (ML Anomaly Pipeline) | Compute Ratio |
|---|---|---|---|
| **10,000 events** | 787 rows/sec (12.70s) | **6,348 rows/sec** (1.58s) | 0.12x (Extension 8.1x faster) |
| **50,000 events** | 18,744 rows/sec (2.67s) | **6,469 rows/sec** (7.73s) | 2.90x |

**Analysis:** On smaller batches (10k), Spark's multi-stage shuffle sort for baseline state grouping incurs significant driver scheduling overhead. On large batches (50k), simple counting is 2.9x faster than evaluating 4 ML detectors, demonstrating the expected compute cost of multivariate anomaly inference in stream processors.

---

## 8. Original Repository File Mapping

| Original File | Role in Baseline | Modifications Made & Rationale | Current Location |
|---|---|---|---|
| `iotsimulator.py` | Python 2 synthetic random generator for state temperatures | Replaced by `src/synthetic_generator.py` and `src/producer.py`. Upgraded to Python 3, added AI4I 2020 empirical dataset ingestion, physics-informed milling machine modeling, and configurable rate-limiting up to 1M events. | Preserved in [`original_repo/iotsimulator.py`](original_repo/iotsimulator.py); replaced by [`src/producer.py`](src/producer.py) |
| `kafka-direct-iotmsg.py` | Legacy Spark 2.0 DStreams consumer printing state averages to stdout | Upgraded to Spark 4.2.0 Structured Streaming. Implemented schema validation, feature engineering, windowed aggregations, multi-detector scoring (Threshold, Z-score, Isolation Forest), and durable Parquet/CSV and Kafka sinks. | Preserved in [`original_repo/kafka-direct-iotmsg.py`](original_repo/kafka-direct-iotmsg.py); replaced by [`src/streaming/spark_stream_processor.py`](src/streaming/spark_stream_processor.py) |
| `README.md` | Basic project tutorial for AWS EC2 Spark 2.0 setup | Expanded into full research documentation with problem formulation, citations, physics formulas, experimental results, and runbooks. | Replaced by this root [`README.md`](README.md) |
| `image.png` | Baseline architecture diagram | Modernized with interactive Mermaid diagram and Matplotlib benchmark plots. | Preserved in [`original_repo/image.png`](original_repo/image.png) |

---

## 9. Quickstart & Execution Guide

### Prerequisites
- Linux or WSL 2 (Ubuntu 22.04 / 24.04 / 26.04)
- Java 21 or Java 17 OpenJDK (`java -version`)
- Python 3.10+ (tested on Python 3.14.4)

### Step 1: Environment Setup
```bash
# Clone and enter directory
cd /home/srajimon/big_data_project

# Activate existing virtualenv or create a new one
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Start Apache Kafka (Standalone KRaft Mode)
No Docker daemon or root privileges required:
```bash
bash scripts/start_kafka.sh
```
*(Alternatively, if Docker Compose is available: `docker compose up -d`)*

### Step 3: Run Offline Training & Holdout Evaluation
```bash
./venv/bin/python -m src.models.train_offline
```

### Step 4: Run Automated Tests
```bash
./venv/bin/pytest tests/ -v
```

### Step 5: Execute End-to-End Streaming Pipeline
```bash
bash scripts/run_pipeline.sh
```

### Step 6: Run Scalability Benchmarks & Generate Plots
```bash
bash scripts/run_benchmarks.sh
```

### Step 7: Launch Interactive Streamlit Dashboard
```bash
bash scripts/run_dashboard.sh
# Open http://localhost:8501 in your browser
```

---

## 10. Practical Limitations & Threats to Validity

1. **Replay Simulation vs Live Factory Sensors:** Replaying historical CSV records through a local Kafka broker is a simulation. It does not reproduce industrial network packet drops, MQTT broker gateways, or clock skew between separate IoT microcontrollers.
2. **Failure Labels vs General Operational Anomalies:** The AI4I dataset's `machine_failure` labels represent catastrophic mechanical stops. Unsupervised algorithms (such as Isolation Forest) may flag true operational outliers that do not ultimately result in a mechanical failure, which penalizes nominal precision.
3. **Local Parallelism vs Physical Clusters:** The pipeline was validated using Spark's `local[*]` executor mode on an 8-core CPU. Running on multi-node physical clusters (e.g., Kubernetes or YARN) introduces network serialization and shuffle partition latency.

---

## 11. Project Structure

```
big_data_project/
├── config/
│   └── kafka_kraft.properties       # Apache Kafka KRaft broker configuration
├── dashboard/
│   └── app.py                       # Interactive Streamlit dashboard
├── data/
│   └── ai4i2020.csv                 # Primary AI4I 2020 Predictive Maintenance CSV
├── docs/
│   ├── architecture.md              # In-depth system architecture documentation
│   ├── experiment_plan.md           # Formal hypotheses, variables, and experiment plan
│   └── presentation_outline.md      # 12-slide academic presentation outline
├── models/
│   ├── detector_metadata.json       # Isolation Forest metadata & thresholds
│   ├── isolation_forest.joblib      # Serialized scikit-learn Isolation Forest model
│   ├── scaler.joblib                # Serialized StandardScaler
│   ├── thresholds.json              # Calibrated domain thresholds
│   └── zscore_params.json           # Normal training-set means and standard deviations
├── original_repo/                   # Preserved original baseline repository (Yugo Kato)
│   ├── iotsimulator.py
│   ├── kafka-direct-iotmsg.py
│   └── README.md
├── output/
│   ├── benchmarks/
│   │   ├── benchmark_results.json   # Machine-readable benchmark outputs
│   │   ├── benchmark_summary.csv    # Scaling benchmark table
│   │   └── plots/                   # Generated Matplotlib publication figures
│   │       ├── baseline_vs_extension.png
│   │       ├── detector_metrics.png
│   │       ├── latency_comparison.png
│   │       └── throughput_scaling.png
│   ├── offline_test_metrics.json    # Holdout test split evaluation metrics
│   ├── offline_test_predictions.csv # Holdout test split predictions
│   └── stream_results/              # Streaming micro-batch CSV and Parquet sinks
├── scripts/
│   ├── run_benchmarks.sh            # Benchmark suite execution script
│   ├── run_dashboard.sh             # Streamlit dashboard launcher
│   ├── run_pipeline.sh              # End-to-end pipeline runner
│   ├── start_kafka.sh               # Kafka KRaft startup script
│   └── stop_kafka.sh                # Kafka broker stop script
├── src/
│   ├── baseline_modern.py           # Modernized baseline execution for fair comparison
│   ├── benchmark.py                 # Repeatable benchmark suite
│   ├── config.py                    # Centralized settings and constants
│   ├── data_loader.py               # AI4I data acquisition, verification, and splits
│   ├── evaluation.py                # Metric calculations (Precision, Recall, ROC-AUC)
│   ├── producer.py                  # High-throughput Kafka producer & replay
│   ├── synthetic_generator.py       # High-volume synthetic telemetry generator
│   ├── models/
│   │   ├── detectors.py             # Modular Threshold, ZScore, and IForest detectors
│   │   └── train_offline.py         # Offline zero-leakage training script
│   └── streaming/
│       └── spark_stream_processor.py# Spark Structured Streaming engine
├── tests/
│   ├── test_detectors.py            # Unit tests for anomaly detectors
│   ├── test_evaluation_metrics.py   # Unit tests for classification metrics
│   ├── test_kafka_spark_integration.py # Kafka & Spark integration tests
│   ├── test_schema_and_features.py  # Unit tests for schema and physics features
│   └── test_serialization_and_replay.py # Unit tests for serialization and generator
├── docker-compose.yml               # Containerized Kafka deployment configuration
├── pytest.ini                       # Test suite runner settings
├── requirements.txt                 # Python project dependencies
└── README.md                        # Project root documentation
```
