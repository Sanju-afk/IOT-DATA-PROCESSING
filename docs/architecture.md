# System Architecture: Real-Time Industrial IoT Anomaly Detection Pipeline

## 1. Overview

This document details the architectural design of the **Real-Time Industrial IoT Anomaly Detection and Scalable Stream Analytics** system, a distributed predictive maintenance and streaming anomaly analytics pipeline.

### 1.1 System Capabilities
- **Runtime:** Python 3.14 and Apache Spark 4.2.0 Structured Streaming.
- **Dataset:** Ingests the empirical **UCI AI4I 2020 Predictive Maintenance Dataset** (and high-fidelity synthetic telemetry for stress benchmarking).
- **Physics-Informed Features:** Computes temperature differential ($\Delta T$), mechanical power ($P = \tau \cdot \omega$), and tool overstrain ($S = W \cdot \tau$).
- **Multi-Method Anomaly Detection:** Implements a uniform detector abstraction supporting:
  1. *Method A:* Physical Domain & Rule-Based Thresholds
  2. *Method B:* Statistical Distance / Standardized Z-Score
  3. *Method C:* Multivariate Tree-Based Isolation Forest
  4. *Method D:* Stacked Ensemble (cost-sensitive Logistic Regression meta-model over the other three detectors' continuous scores, F2-tuned decision threshold)
- **Durable Sinks:** Every scored micro-batch lands in a partitioned Hive table (`iot_analytics.telemetry_scored`, Parquet) for downstream SQL querying, alongside per-batch CSV files; detected anomalies are also published to a dedicated Kafka topic (`iot_anomalies`).
- **Interactive UI:** A real-time and historical analytics dashboard built in Streamlit.

---

## 2. End-to-End Pipeline Topology

The diagram below illustrates the complete data flow from telemetry generation to real-time ingestion, stream processing, anomaly scoring, durable persistence, and visualization.

```mermaid
flowchart TD
    subgraph Data_Source ["1. Telemetry Ingestion Layer"]
        A1["AI4I 2020 Labeled CSV<br/>(10,000 Records)"] --> B["Python Replay & Synthetic Producer<br/>(src/producer.py)"]
        A2["Industrial Synthetic Generator<br/>(Up to 1,000,000 Events)"] --> B
    end

    subgraph Messaging ["2. Distributed Message Bus"]
        B -->|JSON Events| C["Kafka Topic: iot_telemetry<br/>(Port 9092, KRaft Mode)"]
    end

    subgraph Stream_Processing ["3. Distributed Stream Processing (Apache Spark 4.2.0)"]
        C -->|Structured Streaming Source| D["Spark Streaming Ingest<br/>(Micro-batch Interval: 2s)"]
        D --> E["Schema Validation & Filtering<br/>(Rejects Corrupt Telemetry)"]
        E --> F["Physics Feature Engineering<br/>(Power, Temp Diff, Overstrain)"]
        F --> G["Tumbling / Sliding Window Aggregations<br/>(10-second Windows, Watermark: 30s)"]
        F --> H["ForeachBatch Micro-Batch Processor"]
    end

    subgraph Detection_Engine ["4. Anomaly Detection Engine (src/models/detectors.py)"]
        H --> I["Pre-Trained Offline Models<br/>(Zero Leakage Split)"]
        I --> J1["Method A: Domain Thresholds<br/>(Power, Tool Wear, HDF)"]
        I --> J2["Method B: Standardized Z-Score<br/>(RMS & Max Abs Distance)"]
        I --> J3["Method C: Isolation Forest<br/>(Multivariate Tree Splitting)"]
        J1 & J2 & J3 --> K["Method D: Stacked Ensemble<br/>(Logistic Regression over continuous scores)"]
    end

    subgraph Sinks ["5. Sinks & Downstream Integration"]
        K -->|Detected Anomaly Events| L["Kafka Topic: iot_anomalies"]
        K -->|Scored Telemetry Records| W["Hive Table: iot_analytics.telemetry_scored<br/>(Parquet, partitioned by event_date)"]
        K -->|Per-batch Files| M["CSV Sinks<br/>(output/stream_results/)"]
        G -->|Windowed Statistics| N["Kafka Topic: iot_aggregates"]
    end

    subgraph Presentation ["6. Downstream Querying & Dashboard Layer"]
        W --> Q["Hive Views: anomalies, hourly_detector_summary<br/>(Spark SQL / HiveQL via src/warehouse.py)"]
        M --> O["Streamlit Interactive Dashboard<br/>(dashboard/app.py)"]
        L --> O
    end
```

---

## 3. Data Schemas & Physics Specifications

### 3.1 Raw Ingest Schema (`iot_telemetry`)
| Field | Type | Unit | Description |
|---|---|---|---|
| `udi` | Integer | - | Continuous record sequence identifier (1 to 10,000) |
| `product_id` | String | - | Machine product variant identifier (e.g. `L47181`, `M14860`, `H29000`) |
| `product_type` | String | - | Quality variant: `L` (50% low), `M` (30% medium), `H` (20% high) |
| `air_temperature` | Float | Kelvin [K] | Ambient factory temperature ($\approx 298 - 304$ K) |
| `process_temperature` | Float | Kelvin [K] | Internal machining operational temperature ($\approx 308 - 314$ K) |
| `rotational_speed` | Integer | RPM | Spindle angular rotational speed ($\approx 1200 - 2800$ rpm) |
| `torque` | Float | Newton-meters [Nm] | Measured shaft torque ($\approx 10 - 75$ Nm) |
| `tool_wear` | Integer | Minutes [min] | Accumulated cutting tool operational duration |
| `machine_failure` | Integer | Binary {0, 1} | Ground truth label (1 = Failure, 0 = Normal) |
| `twf`, `hdf`, `pwf`, `osf`, `rnf` | Integer | Binary {0, 1} | Individual failure mode trigger indicators |
| `timestamp` | String | ISO-8601 | Event generation timestamp |

### 3.2 Physics-Based Feature Derivations
Following the milling machine operational principles established by Matzka (2020):
1. **Temperature Differential ($\Delta T$):**
   $$\Delta T = T_{\text{process}} - T_{\text{air}}$$
   *Physical Significance:* Under normal dissipation, $\Delta T \ge 8.6\text{ K}$. If $\Delta T < 8.6\text{ K}$ while speed is low ($\le 1380\text{ rpm}$), convective heat transfer fails, triggering **Heat Dissipation Failure (HDF)**.
2. **Mechanical Power ($P$):**
   $$P = \tau \cdot \omega = \tau \cdot \left(\text{speed}_{\text{rpm}} \cdot \frac{2\pi}{60}\right)\quad [\text{Watts}]$$
   *Physical Significance:* Normal milling operations require between $3500\text{ W}$ and $9000\text{ W}$. Values outside this window trigger **Power Failure (PWF)** due to motor stall or extreme overloading.
3. **Overstrain Product ($S$):**
   $$S = W_{\text{tool}} \cdot \tau\quad [\text{min}\cdot\text{Nm}]$$
   *Physical Significance:* High torque acting upon a worn tool causes catastrophic shear fracture, triggering **Overstrain Failure (OSF)** when exceeding $11,000\text{ (L)}$, $12,000\text{ (M)}$, or $13,000\text{ (H)}$ min$\cdot$Nm.

---

## 4. Anomaly Detection Methods

Each detector adheres to an abstract interface (`fit(train_df)`, `predict_record(dict)`, `predict_batch(df)`, `save(path)`, `load(path)`):

### 4.1 Method A: Fixed Domain & Statistical Thresholds (`ThresholdDetector`)
- Uses deterministic physical inequalities derived from equipment operational specifications:
  - $P < 3500\text{ W}$ or $P > 9000\text{ W}$
  - $W_{\text{tool}} \ge 200\text{ min}$
  - $\Delta T < 8.6\text{ K} \land \omega \le 1380\text{ rpm}$
  - $S > S_{\text{limit}}(\text{variant})$
  - Extreme torque ($> 68\text{ Nm}$ or $< 12\text{ Nm}$)
- **Strengths:** 100% explainable, zero training latency, catches 97.1% of machine failure instances.
- **Weaknesses:** Higher false alarm rate (7.8% FPR, 30.6% precision) because operating near extreme boundaries does not always result in catastrophic failure.

### 4.2 Method B: Standardized Z-Score Anomaly Detection (`ZScoreDetector`)
- Computes normalized Euclidean distance from the normal baseline distribution:
  $$z_j = \frac{x_j - \mu_j}{\sigma_j},\quad \text{RMS}(Z) = \sqrt{\frac{1}{d}\sum_{j=1}^d z_j^2}$$
- Anomaly triggered when $\max(|z_j|) \ge 3.0$ or $\text{RMS}(Z) \ge 2.5$.
- **Strengths:** Very low false alarm rate (1.24% FPR, 33.3% precision).
- **Weaknesses:** Fails to detect failures resulting from correlated multivariate shifts where each individual sensor remains within standard 3-sigma limits (Recall = 17.65%).

### 4.3 Method C: Multivariate Isolation Forest (`IsolationForestDetector`)
- Unsupervised recursive tree-partitioning algorithm fit exclusively on normal training data ($\text{machine\_failure} == 0$).
- Detects anomalies based on path length $h(x)$ required to isolate sample $x$:
  $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
- Evaluated on normalized sensor space using `StandardScaler`.
- **Strengths:** Efficiently captures nonlinear multivariate correlations across temperature, speed, torque, and tool wear without assuming normal Gaussian distributions.
- **Performance:** 89.3% ROC-AUC, 36.8% Recall, 3.2% FPR.

### 4.4 Method D: Stacked Ensemble (`UnifiedPipelineDetector.fit_meta`)
- Hard majority voting ($\sum \text{pred}_k \ge 2$) discards each detector's confidence and treats a borderline signal the same as a decisive one, so a strong-but-lone physics signal gets outvoted. `UnifiedPipelineDetector` instead feeds the three detectors' *continuous* scores (`score_threshold`, `score_zscore`, `score_iforest`) into a stacking meta-model.
- A `StandardScaler` + cost-sensitive `LogisticRegression` (`class_weight={0: 1.0, 1: 5.0}`, i.e. a missed failure costs 5x a false alarm) is fit on a validation split carved out of the training data — disjoint from the holdout test set, unlike the normal-only split used for Z-Score/Isolation Forest.
- The decision threshold is swept over $[0.01, 0.99]$ on that validation split to maximize $F_\beta$ with $\beta = 2$ (recall-weighted), rather than assuming $P \ge 0.5$.
- `pred_majority_vote` (hard 2-of-3 vote) and `pred_any` (OR across all three, a high-recall safety net) are retained as diagnostic columns for comparison.
- The meta-features used to fit the Logistic Regression and sweep the F2 threshold come from `build_oof_meta_features()` in [`src/models/train_offline.py`](../src/models/train_offline.py): a 5-fold stratified cross-validation that re-fits fresh base detectors per fold and scores only the held-out fold, so every row's scores come from detectors that never saw it. This exposes the fit to all 271 training-split failures (instead of ~54 from a single validation slice), mirroring how `sklearn.ensemble.StackingClassifier(cv=...)` generates its meta-features internally.
- **Performance:** **91.18% Recall**, **51.24% Precision** (F1 = 0.6561, F2 = 0.7888, FPR = 3.05%, ROC-AUC = 0.9771) on the holdout test split — missing only 6 of 68 failures, versus 41 missed by hard majority voting (Recall 39.71%, Precision 47.37%, F1 0.4320).

---

## 5. Streaming Execution & Scalability Architecture

1. **Spark Structured Streaming Integration:**
   - Ingests raw JSON messages from Kafka using the `org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0` package.
   - Micro-batch processing ensures exactly-once semantics through offset checkpointing (`output/checkpoints/`).
   - Micro-batch scoring avoids expensive in-stream model fitting: pre-trained Isolation Forest and scaler artifacts are loaded into memory once and evaluated vectorially across micro-batch partitions.
2. **Durability & Fault Tolerance:**
   - Enriched events and detector scores are appended to the Hive table `iot_analytics.telemetry_scored` (see Section 6) and written as per-batch CSV files in `output/stream_results/`.
   - Identified anomalies are simultaneously emitted to `iot_anomalies` Kafka topic to facilitate downstream alerting or automated equipment shutdown.

---

## 6. Hive Warehouse (Downstream Querying Layer)

`MicroBatchProcessor.process_batch` calls `append_scored_batch()` from [`src/warehouse.py`](../src/warehouse.py), which casts the scored pandas micro-batch to a fixed schema and runs `insertInto` on a Hive-format Parquet table. The table and views are created idempotently (`CREATE ... IF NOT EXISTS` / `CREATE OR REPLACE VIEW`) at stream start.

| Object | Kind | Contents |
|---|---|---|
| `iot_analytics.telemetry_scored` | Table, `STORED AS PARQUET`, `PARTITIONED BY (event_date DATE)` | Raw sensor fields, physics features, all detector predictions and scores, `event_time` / `kafka_time` / `processed_time` (UTC timestamps), `latency_ms`, and the Spark `batch_id` |
| `iot_analytics.anomalies` | View | Rows where `pred_ensemble = 1` |
| `iot_analytics.hourly_detector_summary` | View | Per hour and `product_type`: event count, labeled failures, rows flagged by each detector, ensemble true positives, mean latency |

Partitioning by event date lets time-bounded queries skip whole directories. Each micro-batch writes one Parquet file per date partition it touches.

**Storage and metastore.** The defaults are a local warehouse directory (`output/hive_warehouse/`) and an embedded Derby metastore (`output/hive_metastore/`). Derby accepts one JVM at a time. When the streaming job holds it (detected through Derby's OS lock on `metastore_db/dbex.lck`), `python -m src.warehouse` switches to snapshot mode: it registers the table's Parquet directory under the same table and view names in a temporary in-memory catalog. Readers never see half-written batches, because Hive stages each insert in a hidden `.hive-staging` directory and moves the files into place on commit. For concurrent readers, or to place the data on HDFS, set `HIVE_METASTORE_URIS` to a standalone Hive Metastore and `HIVE_WAREHOUSE_DIR` to an `hdfs://` path; Spark's built-in Hive 2.3 client and Hadoop 3 client handle both without code changes. Because the table is Hive-format Parquet, HiveServer2, Trino, or Presto can query it through the same metastore.

**Delivery semantics.** The Hive append is at-least-once. If the job crashes after a batch is written but before Spark commits its checkpoint, the batch is replayed with the same `batch_id` and its rows are appended again. Consumers that need exact counts can deduplicate on `(batch_id, event_seq, udi)`.
