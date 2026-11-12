"""Checkpoint/resume behaviour of the real evaluation harness.

The real suite runs against a rate-limited provider, so a partial run must be
resumable instead of restarting from scratch. These tests exercise the
checkpoint helpers without touching the network.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load_module():
    spec = importlib.util.spec_from_file_location(
        "run_real_eval", ROOT / "scripts" / "run_real_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_real_eval"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def real_eval():
    return load_module()


def test_append_and_load_roundtrip(real_eval, tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    record = {"index": 3, "query": "How do I reset my password?", "intent": "account"}
    real_eval.append_checkpoint(path, record)

    loaded = real_eval.load_checkpoints(path)

    assert loaded[3]["query"] == "How do I reset my password?"
    assert loaded[3]["intent"] == "account"


def test_load_ignores_corrupt_trailing_line(real_eval, tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    good = {"index": 1, "query": "q1"}
    path.write_text(
        json.dumps(good) + "\n" + "{not json}\n", encoding="utf-8"
    )

    loaded = real_eval.load_checkpoints(path)

    assert loaded == {1: good}


def test_load_missing_file_returns_empty(real_eval, tmp_path):
    assert real_eval.load_checkpoints(tmp_path / "absent.jsonl") == {}


def test_later_record_wins_for_same_index(real_eval, tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    real_eval.append_checkpoint(path, {"index": 2, "query": "first"})
    real_eval.append_checkpoint(path, {"index": 2, "query": "second"})

    loaded = real_eval.load_checkpoints(path)

    assert loaded[2]["query"] == "second"


def test_checkpoint_records_are_json_serialisable(real_eval):
    record = {
        "index": 0,
        "query": "q",
        "context": ["chunk one"],
        "retrieved_ids": ["doc#chunk-0"],
    }
    assert json.loads(json.dumps(record)) == record
