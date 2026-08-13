---
name: earnings-review
description: 财报点评工作流。触发条件：用户请求"点评 X 股""生成财报点评""写报告"。
---

# 财报点评 Skill

## 触发条件
用户请求对某 A 股上市公司某报告期进行财报点评。

## 工作流

1. **数据采集**：按需调用以下数据工具（不必全调，按问题侧重选取）：
   - `get_company_info` — 公司基本面
   - `get_financials` — 财务三表 + 同比 + 近 8 期趋势（核心）
   - `get_valuation` — 当前估值
   - `get_industry_ranking` — 行业涨跌排名
   - `get_research_reports` — 卖方研报（可选）
   - `get_holder_change` — 股东户数变化
   - `get_dividend_history` — 分红历史
   - `get_fund_flow` — 主力资金动向

2. **多空辩论**（用户要求多空分析时）：在同一轮并行发出两个 task 调用：
   - `task(subagent_type="bull", description="<股票代码 + 报告期 + 具体分析指令>")`
   - `task(subagent_type="bear", description="<股票代码 + 报告期 + 具体分析指令>")`
   收到结果后原样展示 bull/bear 分析，再给出平衡总结。

3. **读取模板**：
   ```
   read_file('.finagent/skills/earnings-review/assets/report-template.md')
   ```

4. **生成报告内容**：按模板六段结构，整合数据工具结果与多空分析。

5. **写入报告文件**：
   ```
   write_file('.finagent/reports/{stock_code}_{period}_点评.md', <完整报告 markdown>)
   ```
   period 去掉非字母数字字符（保留中文）。

## 报告编辑规范

- 编辑现有报告章节：使用 `edit_file` 精准替换章节内容（`old_string` 为章节完整文本）。
- 查看报告：`read_file('.finagent/reports/<filename>.md')`。
- 禁止重写整个文件除非有意重新生成。
- 编辑是覆盖写，无版本保留。

## 数据冲突处理
跨工具数据矛盾时，以 `get_financials` 为准并标注差异。东财源偶发超时属正常，已内置重试。
