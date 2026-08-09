# FinAgent — A股财报点评 Agent

基于 LangGraph + DeepSeek 的 A 股财报点评助手。输入股票代码与报告期，自动拉取多维财务数据、生成结构化财报点评，并支持导出 Markdown 报告。

## 功能

- **8 个数据端点**：公司信息、财务报表、估值、行业排名、研报、股东变动、分红历史、资金流向
- **多源聚合 + 降级**：AkShare / 东方财富 / 新浪 / 腾讯，主源失败自动切换备源
- **交互式 TUI**：基于 Textual 的终端界面，流式输出，工具调用进度可视化
- **报告生成**：一键导出财报点评 Markdown 报告

## 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

配置 API Key：

```bash
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY
```

## 运行

```bash
python -m finagent
```

TUI 内输入股票代码 + 报告期（如 `002415 2024Q3`）开始分析。输入 `/help` 查看命令。

## 测试

```bash
pytest
```

## 技术栈

LangGraph · LangChain · DeepSeek · Textual · AkShare · Rich
