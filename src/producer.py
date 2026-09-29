"""
High-throughput Kafka Producer and Event Replay Engine for Industrial IoT Telemetry.
Streams AI4I 2020 predictive maintenance data or high-volume synthetic events
into Apache Kafka at configurable throughput rates.
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from kafka import KafkaProducer
from kafka.errors import KafkaError

from src.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    TOPIC_TELEMETRY,
    RAW_DATA_PATH
)
from src.data_loader import load_and_preprocess_dataset, get_train_test_split
from src.synthetic_generator import IndustrialTelemetryGenerator, replay_dataframe_generator


class IoTTelemetryProducer:
    """
    Kafka Producer for Industrial IoT Telemetry streams.
    Supports controlled rate limiting or unconstrained maximum throughput streaming.
    """

    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        topic: str = TOPIC_TELEMETRY,
        batch_size: int = 16384,
        linger_ms: int = 5
    ):
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.producer = KafkaProducer(
            bootstrap_servers=[bootstrap_servers],
            key_serializer=lambda k: str(k).encode("utf-8") if k is not None else None,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks=1,
            batch_size=batch_size,
            linger_ms=linger_ms,
            buffer_memory=67108864,  # 64MB buffer
            compression_type="gzip"
        )

    def publish_event(self, event: Dict[str, Any]) -> None:
        key = event.get("product_id", str(event.get("udi", "1")))
        self.producer.send(self.topic, key=key, value=event)

    def stream_events(
        self,
        event_generator,
        total_events: int,
        rate_limit: Optional[float] = None,
        report_interval: int = 5000
    ) -> Dict[str, Any]:
        """
        Streams events to Kafka.
        rate_limit: max events per second (None for unthrottled max speed)
        """
        print(f"Starting Kafka stream to topic '{self.topic}' (Target: {total_events} events)...")
        if rate_limit:
            print(f"Rate limit: {rate_limit:,.0f} events/sec")
        else:
            print("Running unthrottled (maximum throughput benchmark mode)")

        start_time = time.time()
        events_sent = 0
        interval_start = start_time
        interval_sent = 0

        delay_per_event = (1.0 / rate_limit) if (rate_limit and rate_limit > 0) else 0.0

        try:
            for event in event_generator:
                if events_sent >= total_events:
                    break

                self.publish_event(event)
                events_sent += 1
                interval_sent += 1

                # Rate limiting sleep if configured
                if delay_per_event > 0:
                    time.sleep(delay_per_event)

                if events_sent % report_interval == 0 or events_sent == total_events:
                    now = time.time()
                    elapsed = now - interval_start
                    inst_rate = interval_sent / elapsed if elapsed > 0 else 0
                    overall_elapsed = now - start_time
                    overall_rate = events_sent / overall_elapsed if overall_elapsed > 0 else 0
                    print(
                        f"  [Producer] Sent: {events_sent:,}/{total_events:,} | "
                        f"Instant: {inst_rate:,.0f} evt/s | Overall: {overall_rate:,.0f} evt/s"
                    )
                    interval_start = now
                    interval_sent = 0

            # Flush remaining buffer to Kafka
            self.producer.flush()
        except KeyboardInterrupt:
            print("\nProducer interrupted by user.")
        finally:
            end_time = time.time()
            total_duration = end_time - start_time
            throughput = events_sent / total_duration if total_duration > 0 else 0

        summary = {
            "topic": self.topic,
            "events_sent": events_sent,
            "total_duration_sec": round(total_duration, 4),
            "throughput_events_per_sec": round(throughput, 2)
        }
        print(f"Completed streaming: {events_sent:,} events in {total_duration:.2f}s ({throughput:,.1f} events/sec)")
        return summary

    def close(self):
        self.producer.close()


def main():
    parser = argparse.ArgumentParser(description="Industrial IoT Kafka Telemetry Producer")
    parser.add_argument("--topic", default=TOPIC_TELEMETRY, help="Kafka destination topic")
    parser.add_argument("--servers", default=KAFKA_BOOTSTRAP_SERVERS, help="Kafka bootstrap servers")
    parser.add_argument("--events", type=int, default=10000, help="Total events to stream")
    parser.add_argument("--rate", type=float, default=None, help="Rate limit in events/second (None for max speed)")
    parser.add_argument("--mode", choices=["replay", "synthetic"], default="replay", help="Data stream source mode")
    args = parser.parse_args()

    producer = IoTTelemetryProducer(bootstrap_servers=args.servers, topic=args.topic)

    if args.mode == "replay":
        df = load_and_preprocess_dataset(RAW_DATA_PATH)
        _, test_df = get_train_test_split(df)
        generator = replay_dataframe_generator(test_df, total_events=args.events, loop=True)
    else:
        synth_gen = IndustrialTelemetryGenerator(seed=42)
        generator = (
            synth_gen.generate_record(i, timestamp=datetime.now(timezone.utc))
            for i in range(1, args.events + 1)
        )

    try:
        summary = producer.stream_events(
            event_generator=generator,
            total_events=args.events,
            rate_limit=args.rate
        )
        print(json.dumps(summary, indent=2))
    finally:
        producer.close()


if __name__ == "__main__":
    main()
