from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "build_publish_readiness.py"
    spec = importlib.util.spec_from_file_location("build_publish_readiness", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publish_ready_candidate_requires_strength_and_coverage():
    mod = _load_module()
    rows = [
        {
            "domain": "D1",
            "is_complete": True,
            "positive": True,
            "negative": False,
            "strong_win": True,
            "interim_strong": False,
        },
        {
            "domain": "D2",
            "is_complete": True,
            "positive": True,
            "negative": False,
            "strong_win": False,
            "interim_strong": False,
        },
        {
            "domain": "D3",
            "is_complete": True,
            "positive": False,
            "negative": False,
            "strong_win": False,
            "interim_strong": False,
        },
    ]
    out = mod._campaign_decision(rows)
    assert out["decision"] == "publish_ready_candidate"
    assert out["counts"]["domains_complete"] == 3
    assert out["counts"]["domains_strong_win"] == 1


def test_interim_strong_without_completion_is_flagged():
    mod = _load_module()
    rows = [
        {
            "domain": "D1",
            "is_complete": False,
            "positive": True,
            "negative": False,
            "strong_win": False,
            "interim_strong": True,
        },
        {
            "domain": "D2",
            "is_complete": False,
            "positive": True,
            "negative": False,
            "strong_win": False,
            "interim_strong": False,
        },
    ]
    out = mod._campaign_decision(rows)
    assert out["decision"] == "strong_signal_incomplete"
    assert out["counts"]["domains_complete"] == 0
    assert out["counts"]["domains_interim_strong"] == 1


def test_not_ready_when_no_meaningful_signal():
    mod = _load_module()
    rows = [
        {
            "domain": "D1",
            "is_complete": False,
            "positive": False,
            "negative": False,
            "strong_win": False,
            "interim_strong": False,
        },
        {
            "domain": "D2",
            "is_complete": True,
            "positive": False,
            "negative": True,
            "strong_win": False,
            "interim_strong": False,
        },
    ]
    out = mod._campaign_decision(rows)
    assert out["decision"] == "not_ready"
    assert out["counts"]["domains_complete"] == 1
    assert out["counts"]["domains_complete_negative"] == 1
