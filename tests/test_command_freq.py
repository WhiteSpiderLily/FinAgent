"""Tests for command frequency persistence."""
from pathlib import Path

from finagent.command_freq import CommandFreq


def test_missing_file_is_empty(tmp_path):
    cf = CommandFreq(tmp_path / ".finagent.json")
    assert cf.get("clear") == 0


def test_load_existing(tmp_path):
    p = tmp_path / ".finagent.json"
    p.write_text('{"command_freq": {"clear": 5, "help": 2}}', encoding="utf-8")
    cf = CommandFreq(p)
    assert cf.get("clear") == 5
    assert cf.get("help") == 2


def test_corrupt_json_is_empty(tmp_path):
    p = tmp_path / ".finagent.json"
    p.write_text("not json{{{", encoding="utf-8")
    cf = CommandFreq(p)
    assert cf.get("clear") == 0


def test_non_int_value_skipped(tmp_path):
    p = tmp_path / ".finagent.json"
    p.write_text('{"command_freq": {"clear": 5, "bad": "x"}}', encoding="utf-8")
    cf = CommandFreq(p)
    assert cf.get("clear") == 5
    assert cf.get("bad") == 0


def test_increment_and_save_roundtrip(tmp_path):
    p = tmp_path / ".finagent.json"
    cf = CommandFreq(p)
    cf.increment("clear")
    cf.increment("clear")
    cf.increment("help")
    cf.save()
    cf2 = CommandFreq(p)
    assert cf2.get("clear") == 2
    assert cf2.get("help") == 1


def test_save_preserves_other_keys(tmp_path):
    p = tmp_path / ".finagent.json"
    p.write_text('{"other_config": 42}', encoding="utf-8")
    cf = CommandFreq(p)
    cf.increment("clear")
    cf.save()
    import json
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["other_config"] == 42
    assert data["command_freq"]["clear"] == 1


def test_missing_command_freq_key_is_empty(tmp_path):
    p = tmp_path / ".finagent.json"
    p.write_text('{"other": 1}', encoding="utf-8")
    cf = CommandFreq(p)
    assert cf.get("clear") == 0
