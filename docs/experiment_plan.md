# Scientific Experiment & Benchmark Plan

## 1. Research Problem & Core Question

**Problem:** Industrial machines operate under multivariate thermal, kinematic, and mechanical regimes. Univariate static thresholds and periodic batch evaluations delay detection and either miss multi-variable interactions or flood operators with false alarms. Distributed stream processing platforms (Apache Kafka + Apache Spark) offer horizontal scalability, but how does detection quality trade off against stream ingestion throughput and micro-batch latency?

**Research Question:**
> *How do multivariate anomaly detection methods (Isolation Forest) compare with univariate physical and statistical thresholding in detection quality (Precision, Recall, F1, FPR), and how does the Kafka–Spark streaming pipeline behave as input event volume and throughput scale up to 100,000 events?*

---

## 2. Formal Hypotheses

- **Hypothesis 1 ($H_1$ — Detection Quality Trade-off):**
  * *H1a (Domain Thresholds):* Physical rule-based thresholding will achieve the highest recall ($\ge 90\%$) against labeled machine failures due to capturing known failure mechanics, but will exhibit low precision ($< 35\%$) due to elevated false positive alarms at non-failure operational boundaries.
  * *H1b (Univariate Z-Scores):* Standardized univariate Z-score distance will suffer from low recall ($< 25\%$) because multivariate anomalies (e.g., low temperature differential combined with low speed) frequently fall well within standard 3-sigma single-sensor bounds.
  * *H1c (Multivariate Isolation Forest):* Tree-based multivariate isolation will yield a superior ROC-AUC ($\ge 85\%$) and lower False Positive Rate ($\le 3.5\%$) compared to univariate thresholding.
  * *H1d (Consensus Ensemble):* A majority-vote ensemble will maximize precision ($> 45\%$) and minimize false alarms ($< 2\%$) by filtering out single-detector spurious triggers.

- **Hypothesis 2 ($H_2$ — Stream Scalability & Backpressure):**
  * The Kafka–Spark streaming pipeline will sustain linear or sub-linear processing duration scaling across workloads from 10,000 to 100,000 events, achieving steady-state processing throughput exceeding $4,000\text{ rows/sec}$ in a resource-constrained WSL environment.

---

## 3. Experimental Variables

### 3.1 Independent Variables (Manipulated Factors)
1. **Workload Scale (Event Count $N$):** $10,000$, $50,000$, and $100,000$ events.
2. **Ingestion Mode:**
   - *Throttled Stream:* Fixed rate limit ($1,000\text{ events/sec}$) simulating steady industrial sensor cadence.
   - *Unthrottled Burst:* Maximum unconstrained producer throughput to stress-test broker and consumer buffer capacities.
3. **Analytical Detection Algorithm:**
   - Method A: Physical Domain Thresholding
   - Method B: Standardized Z-Score Distance ($Z \ge 3.0$)
   - Method C: Multivariate Isolation Forest ($100$ estimators, contamination $3.5\%$)
   - Method D: Consensus Ensemble Majority Vote ($\ge 2$ votes)

### 3.2 Dependent Variables (Measured Outcomes)
1. **Producer Throughput:** Events dispatched per second ($\text{evt/s}$).
2. **Processing Throughput:** Micro-batch rows scored and persisted per second ($\text{rows/s}$).
3. **End-to-End Latency:** Time elapsed between event generation timestamp and Spark persistence completion (Mean, Median, 95th percentile, 99th percentile in milliseconds).
4. **Detection Quality Metrics:**
   - True Positives ($TP$), False Positives ($FP$), True Negatives ($TN$), False Negatives ($FN$)
   - Precision: $\frac{TP}{TP + FP}$
   - Recall (Sensitivity): $\frac{TP}{TP + FN}$
   - F1-Score: $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
   - False Positive Rate (FPR): $\frac{FP}{FP + TN}$
   - Area Under ROC Curve (ROC-AUC) & Precision-Recall Curve (PR-AUC)
5. **System Resource Utilization:** Resident Memory Footprint (RAM in MB) and CPU load.

### 3.3 Control Variables (Kept Constant)
- Dataset: Primary AI4I 2020 Predictive Maintenance split (8,000 train, 2,000 test holdout).
- Training Seed: Deterministic pseudo-random seed (`RANDOM_SEED = 42`).
- Kafka Broker Configuration: 3 partitions, replication factor 1, KRaft consensus.
- Spark Execution: Local parallel executor (`local[*]`, 4 shuffle partitions, 2GB driver heap).
- Feature Representation: Scaled numerical telemetry (temperatures, speed, torque, tool wear, power, temp diff).

---

## 4. Hardware & Software Testbed Configuration

- **Host Environment:** Linux WSL 2 on x86_64 CPU (8 logical cores).
- **Available System Memory:** 7.6 GB RAM.
- **Java Virtual Machine:** OpenJDK 21.0.12.1 64-Bit Server VM.
- **Python Runtime:** Python 3.14.4.
- **Apache Kafka:** Version 4.1.2 running in native KRaft standalone mode.
- **Apache Spark:** Version 4.2.0 Structured Streaming with Kafka 0-10 connector.
- **Machine Learning Libraries:** `scikit-learn 1.9.1`, `pandas 2.3.3`, `numpy 2.5.3`.

---

## 5. Repeatable Execution Procedures

### Step 1: Pre-training Offline Detectors & Verification
Fit all models strictly on normal training records and calculate holdout evaluation metrics:
```bash
./venv/bin/python -m src.models.train_offline
```

### Step 2: Running Automated Unit & Integration Tests
Verify zero data leakage, schema validity, and Kafka communication:
```bash
./venv/bin/pytest tests/ -v
```

### Step 3: Executing Multi-Scale Big Data Benchmarks
Execute the full benchmark suite across 10,000, 50,000, and 100,000 events, baseline comparison, and figure generation:
```bash
./venv/bin/python -m src.benchmark --scales 10000 50000 100000
```

### Step 4: Visualizing Results in Interactive Dashboard
Launch Streamlit for interactive real-time and benchmark analysis:
```bash
./venv/bin/streamlit run dashboard/app.py --server.port 8501
```
