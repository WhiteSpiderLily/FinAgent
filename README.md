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

## Skills

FinAgent 支持用户自定义 skill(类似 Claude Code 的 skill 机制)。Skill 是
一组按场景激活的指令,放在 `skill.md` 文件中。

### Skill 目录

FinAgent 扫描两个目录,同名时**项目目录覆盖用户目录**:

- 用户全局: `~/.finagent/skills/<name>/skill.md`
- 项目本地: `./.finagent/skills/<name>/skill.md`

### Skill 文件格式

```markdown
---
name: my-skill
description: 一句话描述,出现在每轮 system-reminder 的 catalog 中
---

# My Skill

激活此 skill 时,你应:
1. ...
2. 用 read_file 加载 assets/template.md
```

`frontmatter` 仅需 `name`(合法字符 `[A-Za-z0-9_-]+`)和 `description`。

保留名(不可用作 skill name): `report` / `clear` / `help` / `quit` / `reload_skills`。

### 激活 skill

- 用户输入 `/<skill-name>`(如 `/my-skill`)
- 或让 agent 调用 `load_skill(name="my-skill")` 工具
- 两种方式产出等价的对话历史

### 渐进式披露

skill.md 仅包含指令;大型资源(模板、脚本)放在 skill 目录的 `assets/` /
`scripts/` 子目录下,agent 按需通过 `read_file(path="skills/my-skill/assets/x.md")`
加载。

### 热更新

新增/修改/删除 skill 文件后,在 TUI 中输入 `/reload_skills` 重新扫描目录。

### 报告路径

生成的财报点评报告写入 `./.finagent/reports/`(项目本地)。
