"""Tests for InputHistory — pure logic, no Textual event loop."""
from finagent.input_history import InputHistory


def test_append_save_load_roundtrip(tmp_path):
    path = tmp_path / "input_history.json"
    h1 = InputHistory(path)
    h1.append("分析000001")
    h1.append("分析000002")
    h1.save()

    h2 = InputHistory(path)
    assert h2.items == ["分析000001", "分析000002"]


def test_consecutive_dedup(tmp_path):
    h = InputHistory(tmp_path / "ih.json")
    h.append("same")
    h.append("same")
    h.append("different")
    h.append("different")
    h.append("same")  # non-consecutive duplicate — kept
    assert h.items == ["same", "different", "same"]


def test_fifo_cap_100(tmp_path):
    h = InputHistory(tmp_path / "ih.json")
    for i in range(105):
        h.append(f"msg{i}")
    assert len(h.items) == 100
    assert h.items[0] == "msg5"      # first 5 dropped
    assert h.items[-1] == "msg104"


def test_load_nonexistent_file_returns_empty(tmp_path):
    h = InputHistory(tmp_path / "nonexistent.json")
    assert h.items == []


def test_corrupt_file_does_not_crash(tmp_path):
    path = tmp_path / "ih.json"
    path.write_text("{ broken json", encoding="utf-8")
    h = InputHistory(path)
    assert h.items == []
