测试时避免并行（可能导致内存爆炸）
Agent tool 存在且可用，用 select:Agent 加载，避免内联执行
最多同时2个subagents工作（避免限速）关闭一个后再spawn下一个
tool result 或者 user prompt 大于300行时，使用 mcp__headroom_compress 工具压缩，需要获取压缩前的结果时，使用 mcp__headroom_retrieve
项目环境是 .venv
@code-review-graph.md
