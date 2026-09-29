# Presentation Outline: Real-Time Industrial IoT Anomaly Detection Using Kafka and Spark

*A 12-Slide Academic Presentation on Distributed Stream Processing, Physics-Informed Features, and Multivariate Anomaly Detection.*

---

## Slide 1: Title & Overview
- **Title:** Real-Time Industrial IoT Anomaly Detection and Scalable Stream Analytics Using Apache Kafka and Apache Spark
- **Authors:** Senior Data Engineering & Applied Research Team
- **Focus Area:** Distributed Systems, Stream Processing, Machine Learning, Predictive Maintenance
- **Core Message:** Combining physics-grounded threshold rules with multivariate Isolation Forest in a modern Kafka–Spark Structured Streaming pipeline enables high-throughput, low-latency failure detection with robust false-alarm filtering.

---

## Slide 2: Industrial Problem & Motivation
- **The Challenge:** High-throughput modern manufacturing equipment (milling, turning, grinding) generates high-frequency multivariate telemetry across temperatures, motor rotational speed, torque, and tool wear.
- **The Failure of Batch Processing:** Batch analytics delays detection until hours after telemetry is collected, leading to catastrophic tool failure, damaged workpieces, and expensive downtime.
- **The Pitfall of Univariate Thresholds:** Independent static thresholds trigger either excessive false alarms or completely miss dangerous multi-variable interactions (e.g., motor stall at moderate temperatures).

---

## Slide 3: Baseline Starting Point (Attribution & Analysis)
- **Starting Repository:** *Spark and Kafka IoT Data Processing and Analytics* (by Yugo Kato).
- **Baseline Capabilities:**
  - Ingested single-variable ambient temperature data mapped to US states.
  - Legacy Spark 2.0 DStreams (`KafkaUtils.createDirectStream`) and Python 2.7.
  - Calculated 4 basic aggregations (state average temperature, message counts, device counts).
- **Identified Limitations:**
  - Deprecated legacy DStream API (incompatible with modern Spark 3/4).
  - Lack of multivariate telemetry, failure labels, and anomaly detection algorithms.
  - No durable persistence (output to console stdout only).

---

## Slide 4: Proposed Extension & Research Questions
- **Research Question:**
  > *How do multivariate anomaly detection methods compare with simple threshold-based detection in detection quality and processing latency, and how does the Kafka–Spark pipeline behave as input event volume and stream throughput increase?*
- **Our Contributions:**
  1. Complete architectural modernization to **Apache Kafka 4.1.2 (KRaft)** and **Apache Spark 4.2.0 Structured Streaming**.
  2. Integration of the empirical **UCI AI4I 2020 Predictive Maintenance Dataset** with zero-leakage training.
  3. Implementation of **physics-based derived features** ($\Delta T$, mechanical power, tool overstrain).
  4. Multi-detector benchmarking (Domain Thresholds vs Z-Score vs Isolation Forest vs Consensus Ensemble).
  5. End-to-end scalability experiments across $10\text{k}$, $50\text{k}$, and $100\text{k}$ events.

---

## Slide 5: Dataset & Industrial Physics Modeling
- **Dataset:** UCI AI4I 2020 Predictive Maintenance (Matzka, 2020). 10,000 instances, 14 attributes, 3.39% ground-truth machine failure rate.
- **Physics Equations Modeled:**
  - **Heat Dissipation Failure (HDF):** Occurs when convective cooling fails:
    $$\Delta T = T_{\text{process}} - T_{\text{air}} < 8.6\text{ K}\quad \land \quad \text{Speed} \le 1380\text{ rpm}$$
  - **Power Failure (PWF):** Mechanical power bounds:
    $$P = \tau \cdot \left(\text{rpm} \cdot \frac{2\pi}{60}\right)\quad \notin [3500\text{ W}, 9000\text{ W}]$$
  - **Tool Wear Failure (TWF):** Progressive abrasive wearout:
    $$W_{\text{tool}} \ge 200\text{ min}$$
  - **Overstrain Failure (OSF):** Product of cutting force and tool wear exceeds variant limit:
    $$S = W_{\text{tool}} \cdot \tau > S_{\text{limit}}(\text{variant})$$

---

## Slide 6: End-to-End Pipeline Architecture
- **Ingestion:** Python Kafka Producer streaming AI4I replay records and high-volume synthetic events.
- **Message Broker:** Apache Kafka 4.1.2 running KRaft metadata mode on port 9092 (3 partitions).
- **Stream Processing:** Apache Spark 4.2.0 Structured Streaming:
  - Ingestion from topic `iot_telemetry`.
  - Schema parsing and invalid record rejection.
  - Streaming SQL feature transformations.
  - Micro-batch scoring with pre-trained offline models.
- **Dual Sinks:**
  - Real-time anomaly alerts emitted to Kafka topic `iot_anomalies`.
  - Enriched event micro-batches persisted to Parquet and CSV in `output/stream_results/`.
- **Dashboard:** Interactive Streamlit web UI for live stream monitoring and empirical reporting.

---

## Slide 7: Anomaly Detection Methods
- **Method A (Domain Thresholds):** Physical boundaries based on Matzka (2020) physics. Deterministic and 100% explainable.
- **Method B (Standardized Z-Score):** Distance from normal operating distribution:
  $$z = \frac{x - \mu}{\sigma},\quad \text{RMS}(Z) = \sqrt{\frac{1}{d}\sum z_j^2}$$
- **Method C (Isolation Forest):** Unsupervised recursive partition trees fit strictly on normal training data ($\text{machine\_failure} == 0$). Identifies multivariate outliers via short average path length.
- **Method D (Consensus Ensemble):** Majority vote requiring $\ge 2$ agreeing detectors to minimize false alarms.

---

## Slide 8: Detection Quality & Class Imbalance Results
*Evaluated on strictly holdout test split (2,000 samples, 68 true failures = 3.4% prevalence).*

| Method | Precision | Recall | F1-Score | FPR | ROC-AUC | Key Behavior |
|---|---|---|---|---|---|---|
| **A. Domain Thresholds** | 30.56% | **97.06%** | **0.4648** | 7.76% | **0.9790** | Catches 66/68 failures; 150 false alarms |
| **B. Z-Score Distance** | 33.33% | 17.65% | 0.2308 | **1.24%** | 0.8666 | Lowest false alarms; misses coupled anomalies |
| **C. Isolation Forest** | 28.74% | 36.76% | 0.3226 | 3.21% | 0.8926 | Captures non-linear multivariate interactions |
| **D. Consensus Ensemble** | **47.37%** | 39.71% | 0.4320 | 1.55% | N/A | **Highest precision (47.4%)** & lowest false alarms |

- **Critical Insight:** In severe class imbalance (3.4%), naive accuracy is 96.6% but completely useless. Evaluating Precision, Recall, and FPR reveals the true operational trade-offs.

---

## Slide 9: Big Data Scalability Experiments
*Empirical benchmarks across scaling workloads on 8-core WSL with 7.6GB RAM:*

| Workload ($N$) | Producer Throughput | Spark Processing Throughput | Processing Duration | 95th % Latency | Resident Memory |
|---|---|---|---|---|---|
| **10,000 events** | 4,037 evt/sec | 4,631 rows/sec | 2.16 sec | 89.7 sec* | 342.8 MB |
| **50,000 events** | 4,429 evt/sec | **6,509 rows/sec** | 7.68 sec | 454.5 sec* | 425.8 MB |
| **100,000 events** | 4,264 evt/sec | 6,286 rows/sec | 15.91 sec | 907.6 sec* | 524.5 MB |

- *Steady-State Throughput:* Sustains **$6,200 - 6,500\text{ rows/sec}$** across all micro-batches.
- *Memory Efficiency:* Peak memory remains under $525\text{ MB}$, well within standard industrial edge appliance constraints.
- *(Note on Latency:* Latency measures total wall-clock lag from event generation to end of unthrottled burst processing).*

---

## Slide 10: Baseline vs Extension Workload Comparison
- **Original Baseline:** Simple state-level group-by aggregations and distinct counts over single-variable temperature.
- **Proposed Extension:** Full multivariate ingestion, derived physics calculations, and 4 concurrent anomaly detectors.
- **Benchmark Findings:**
  - At **10,000 events**: Extension runs at **$6,348\text{ rows/sec}$** vs Baseline's **$787\text{ rows/sec}$** ($8.1\times$ speedup), because Spark's multi-stage shuffle sort in the baseline incurs high overhead on small batches.
  - At **50,000 events**: Simple aggregations scale to $18,744\text{ rows/sec}$, while the multivariate ML pipeline achieves $6,469\text{ rows/sec}$ ($2.9\times$ compute ratio).
  - *Trade-off:* ML detection requires $2.9\times$ more CPU per event than simple counting, but delivers actionable predictive maintenance intelligence.

---

## Slide 11: Limitations & Threats to Validity
1. **Replay vs Live Telemetry:** Replaying a static CSV in simulated time does not account for factory network jitter, out-of-order packet drops, or transient wireless disconnects.
2. **Failure Labels as Ground Truth:** In industrial reality, machine failures are not exhaustive ground truth for every anomalous sensor deviation (equipment can operate abnormally without catastrophic failure).
3. **Local WSL Environment:** Spark was run in `local[*]` mode. A multi-node physical cluster would introduce inter-node serialization and network shuffle overhead.
4. **Cold Start & Window Warmup:** Rolling window statistics require an initial warm-up period before reliable baselines are established.

---

## Slide 12: Conclusions & Key Takeaways
1. **Architecture Modernization:** Migrating from legacy DStreams to Spark Structured Streaming and Kafka KRaft delivers a reliable, fault-tolerant industrial streaming pipeline.
2. **Domain Rules vs Multivariate ML:**
   - Physical domain thresholds achieve near-perfect recall (97.1%) but higher false alarms.
   - Isolation Forest captures subtle multi-sensor anomalies with low false alarms (3.2%).
   - Consensus ensembles achieve the highest operational precision (47.4%).
3. **Scalability:** The pipeline sustains $> 6,200\text{ events/sec}$ throughput on standard commodity hardware with minimal memory footprint ($< 525\text{ MB}$).
4. **Open Source Deliverables:** Fully runnable scripts, automated benchmarks, unit tests, and Streamlit dashboard ready for reproduction.
