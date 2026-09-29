"""
Integration tests for Kafka publish, consumption, and streaming detector pipeline.
"""

import json
import time
import pytest
from kafka import KafkaProducer, KafkaConsumer

from src.config import KAFKA_BOOTSTRAP_SERVERS
from src.streaming.spark_stream_processor import load_offline_detectors


def test_kafka_publish_and_consume():
    test_topic = "test_integration_topic"
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    test_payload = {
        "udi": 99999,
        "product_id": "L99999",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 10
    }

    future = producer.send(test_topic, value=test_payload)
    record_meta = future.get(timeout=10)
    assert record_meta.topic == test_topic
    producer.close()

    consumer = KafkaConsumer(
        test_topic,
        bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
        auto_offset_reset="earliest",
        consumer_timeout_ms=5000,
        value_deserializer=lambda m: json.loads(m.decode("utf-8"))
    )

    received = False
    for msg in consumer:
        if msg.value.get("udi") == 99999:
            received = True
            break
    consumer.close()
    assert received is True


def test_offline_detector_scoring():
    detector = load_offline_detectors()
    sample_rec = {
        "udi": 1,
        "product_id": "L12345",
        "product_type": "L",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 20
    }
    scored = detector.predict_record(sample_rec)
    assert "pred_threshold" in scored
    assert "pred_zscore" in scored
    assert "pred_iforest" in scored
    assert "pred_ensemble" in scored
