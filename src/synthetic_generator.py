"""
Synthetic industrial IoT telemetry generator and stream replay engine.
Supports high-volume event generation (up to 1,000,000 events) for scalability benchmarking,
closely modeling the multivariate distributions, sensor physics, and failure modes of AI4I 2020.
"""

import datetime
import math
import random
from typing import Dict, Any, Generator, Optional
import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, DOMAIN_THRESHOLDS


class IndustrialTelemetryGenerator:
    """
    Generates synthetic industrial telemetry modeled after milling machine physics.
    Simulates:
    - Normal operation: rotational speed ~1500 rpm, torque ~40 Nm, temperatures ~300K,
      tool wear progressing linearly with stochastic increments.
    - Failure modes:
      * TWF: tool wear > 200 min
      * HDF: low temperature difference (< 8.6 K) at low speed (< 1380 rpm)
      * PWF: mechanical power < 3500 W or > 9000 W
      * OSF: tool_wear * torque > variant threshold
      * RNF: 0.1% independent random anomalies
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        self.tool_wear_counters = {"L": 0.0, "M": 0.0, "H": 0.0}

    def generate_record(
        self,
        udi: int,
        timestamp: Optional[datetime.datetime] = None,
        force_failure_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generates a single synthetic industrial sensor telemetry event."""
        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc)

        # Machine variant (L: 50%, M: 30%, H: 20%)
        variant_prob = self.rng.random()
        if variant_prob < 0.5:
            product_type = "L"
            product_id = f"L{47000 + (udi % 10000):05d}"
        elif variant_prob < 0.8:
            product_type = "M"
            product_id = f"M{14000 + (udi % 10000):05d}"
        else:
            product_type = "H"
            product_id = f"H{29000 + (udi % 10000):05d}"

        # Ambient air temperature [K] ~ N(300, 2)
        air_temp = float(self.rng.normal(300.0, 2.0))
        # Normal process temp difference ~ 10.0 to 11.5 K
        normal_temp_diff = float(self.rng.normal(10.7, 1.0))
        process_temp = air_temp + normal_temp_diff

        # Operating speed [rpm] ~ N(1538, 179)
        rotational_speed = float(self.rng.normal(1538.0, 179.0))
        # Torque inversely related to speed around ~40 Nm
        torque = float(40.0 - 0.05 * (rotational_speed - 1538.0) + self.rng.normal(0, 5.0))
        torque = max(10.0, min(80.0, torque))

        # Tool wear accumulation
        self.tool_wear_counters[product_type] += float(self.rng.uniform(1.0, 3.0))
        if self.tool_wear_counters[product_type] > 250.0:
            self.tool_wear_counters[product_type] = 0.0  # tool replaced
        tool_wear = self.tool_wear_counters[product_type]

        twf = 0
        hdf = 0
        pwf = 0
        osf = 0
        rnf = 0

        # Inject or naturally evaluate failure modes
        if force_failure_type == "TWF" or (tool_wear > 210 and self.rng.random() < 0.4):
            twf = 1
            tool_wear = max(tool_wear, 215.0)

        temp_diff = process_temp - air_temp
        if force_failure_type == "HDF" or (temp_diff < 8.6 and rotational_speed < 1380.0):
            hdf = 1
        elif self.rng.random() < 0.008:  # 0.8% chance to trigger HDF condition
            air_temp = float(self.rng.normal(303.0, 1.0))
            process_temp = air_temp + float(self.rng.uniform(6.5, 8.4))
            rotational_speed = float(self.rng.uniform(1150.0, 1370.0))
            hdf = 1

        power = torque * rotational_speed * (2.0 * math.pi / 60.0)
        if force_failure_type == "PWF" or power < 3500.0 or power > 9000.0:
            pwf = 1
        elif self.rng.random() < 0.007:  # 0.7% chance PWF
            if self.rng.random() < 0.5:
                torque = float(self.rng.uniform(65.0, 78.0))
                rotational_speed = float(self.rng.uniform(1600.0, 2000.0))
            else:
                torque = float(self.rng.uniform(12.0, 18.0))
                rotational_speed = float(self.rng.uniform(1200.0, 1400.0))
            pwf = 1

        overstrain_limit = DOMAIN_THRESHOLDS["overstrain_limits"][product_type]
        if force_failure_type == "OSF" or (tool_wear * torque > overstrain_limit):
            osf = 1
        elif self.rng.random() < 0.007:  # 0.7% chance OSF
            tool_wear = float(self.rng.uniform(180.0, 230.0))
            torque = float(self.rng.uniform(58.0, 72.0))
            osf = 1

        if force_failure_type == "RNF" or self.rng.random() < 0.001:  # 0.1% random failure
            rnf = 1

        machine_failure = 1 if (twf or hdf or pwf or osf or rnf) else 0

        # Round values for sensor realism
        return {
            "udi": int(udi),
            "product_id": str(product_id),
            "product_type": str(product_type),
            "air_temperature": round(float(air_temp), 1),
            "process_temperature": round(float(process_temp), 1),
            "rotational_speed": int(round(rotational_speed)),
            "torque": round(float(torque), 1),
            "tool_wear": int(round(tool_wear)),
            "machine_failure": int(machine_failure),
            "twf": int(twf),
            "hdf": int(hdf),
            "pwf": int(pwf),
            "osf": int(osf),
            "rnf": int(rnf),
            "timestamp": timestamp.isoformat()
        }

    def generate_batch(self, count: int, start_udi: int = 1) -> pd.DataFrame:
        """Generates a batch of records as a Pandas DataFrame."""
        records = []
        base_time = datetime.datetime.now(datetime.timezone.utc)
        for i in range(count):
            event_time = base_time + datetime.timedelta(milliseconds=i * 10)
            records.append(self.generate_record(start_udi + i, timestamp=event_time))
        return pd.DataFrame(records)


def replay_dataframe_generator(
    df: pd.DataFrame,
    total_events: Optional[int] = None,
    loop: bool = True
) -> Generator[Dict[str, Any], None, None]:
    """
    Replays records from an existing DataFrame (e.g. AI4I test split),
    injecting dynamic timestamps while preserving source telemetry and labels.
    """
    n_rows = len(df)
    events_yielded = 0
    idx = 0
    base_time = datetime.datetime.now(datetime.timezone.utc)

    while True:
        if total_events is not None and events_yielded >= total_events:
            break
        if idx >= n_rows:
            if not loop:
                break
            idx = 0  # loop back

        row = df.iloc[idx].to_dict()
        current_time = base_time + datetime.timedelta(milliseconds=events_yielded * 10)
        row["timestamp"] = current_time.isoformat()
        row["event_seq"] = events_yielded

        # Clean NaN if present
        cleaned_row = {
            k: (int(v) if isinstance(v, (np.integer, int)) else
                float(v) if isinstance(v, (np.floating, float)) and not math.isnan(v) else
                str(v) if isinstance(v, str) else v)
            for k, v in row.items()
        }

        yield cleaned_row
        events_yielded += 1
        idx += 1
