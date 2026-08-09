"""tests/test_report.py"""
from unittest.mock import patch, MagicMock
from finagent.report import generate_report, _messages_to_text


_FAKE_MESSAGES = [
    {"role": "user", "content": "分析 002415 2024Q3 财报"},
    {"role": "assistant", "content": "海康威视 2024三季报：营收 650亿..."},
]


def test_generate_report_returns_filepath(tmp_path):
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="# 海康威视(002415) 2024Q3财报点评\n\n## 一、事件概述\n...")
    with patch("finagent.report.get_llm", return_value=fake_llm), \
         patch("finagent.report.Path.cwd", return_value=tmp_path):
        filepath, content = generate_report(_FAKE_MESSAGES)
    assert "002415" in filepath
    assert "2024Q3" in filepath
    assert ".md" in filepath
    assert "事件概述" in content


def test_generate_report_error_on_missing_info():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="[ERROR] 无法从对话历史确定公司代码或报告期")
    with patch("finagent.report.get_llm", return_value=fake_llm):
        try:
            generate_report([{"role": "user", "content": "你好"}])
            assert False, "should have raised"
        except ValueError:
            pass


def test_messages_to_text_handles_langchain_types():
    text = _messages_to_text([
        {"role": "human", "content": "test question"},
        {"role": "ai", "content": "test answer"},
    ])
    assert "分析师" in text
    assert "助手" in text
    assert "[human]" not in text
    assert "[ai]" not in text
