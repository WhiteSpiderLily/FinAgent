<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<br />
<div align="center">

  <h3 align="center">FinAgent</h3>

  <p align="center">
    A股财报点评 Agent — 基于 LangGraph + Deep Agents + DeepSeek 的 A 股财报点评助手
    <br />
    <a href="https://github.com/WhiteSpiderLily/FinAgent/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/WhiteSpiderLily/FinAgent/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a>
      <ul>
        <li><a href="#记忆">记忆</a></li>
        <li><a href="#skills">Skills</a></li>
      </ul>
    </li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

输入股票代码与报告期，自动拉取多维财务数据、生成结构化财报点评，并支持导出 Markdown 报告。

- **8 个数据端点**：公司信息、财务报表、估值、行业排名、研报、股东变动、分红历史、资金流向
- **多源聚合 + 降级**：AkShare / 东方财富 / 新浪 / 腾讯，主源失败自动切换备源
- **多空辩论 subagents**：基于 Deep Agents 的 bull/bear 对抗分析
- **交互式 TUI**：基于 Textual 的终端界面，流式输出，工具调用进度可视化
- **输入增强**：多行输入、历史上下翻、斜杠命令/技能自动补全（模糊匹配 + 频率排序）
- **会话持久化**：退出自动保存，`--resume <session_id>` 恢复历史会话
- **长期记忆**：用户/项目级记忆文件 + 自动记忆提取与治理（`.finagent/memory/`）
- **报告生成**：一键导出财报点评 Markdown 报告

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [Python](https://www.python.org/) (>= 3.11)
* [LangGraph](https://github.com/langchain-ai/langgraph)
* [LangChain](https://github.com/langchain-ai/langchain)
* [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
* [DeepSeek](https://www.deepseek.com/)
* [Textual](https://github.com/Textualize/textual)
* [AkShare](https://github.com/akfamily/akshare)
* [Rich](https://github.com/Textualize/rich)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Python >= 3.11
* DeepSeek API Key

### Installation

1. 获取 DeepSeek API Key：<https://platform.deepseek.com>
2. Clone the repo
   ```sh
   git clone https://github.com/WhiteSpiderLily/FinAgent.git
   ```
3. 创建虚拟环境并安装依赖
   ```sh
   cd FinAgent
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   ```
4. 配置 API Key
   ```sh
   cp .env.example .env
   # 编辑 .env 填入 DEEPSEEK_API_KEY
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

```sh
python -m finagent
```

TUI 内输入股票代码 + 报告期（如 `002415 2024Q3`）开始分析。输入 `/help` 查看命令。

恢复上次会话：

```sh
python -m finagent --resume <session_id>
```

运行测试：

```sh
pytest
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### 记忆

启动时注入以下记忆文件（存在才加载，变更才重新注入）：

- 用户级：`~/.finagent/finagent.md`
- 项目级：`.finagent/finagent.md`
- 自动记忆摘要：`.finagent/memory/memory.md`

agent 每轮自动提取有价值信息写入 `.finagent/memory/`（preference / project / feedback / reference 分文件），`memory.md` 仅由治理流程维护（每 24h 且满 5 个 session 触发整理压缩）。

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Skills

FinAgent 支持用户自定义 skill（类似 Claude Code 的 skill 机制）。Skill 是
一组按场景激活的指令，放在 `skill.md` 文件中。

#### Skill 目录

FinAgent 扫描两个目录，同名时**项目目录覆盖用户目录**：

- 用户全局：`~/.finagent/skills/<name>/skill.md`
- 项目本地：`./.finagent/skills/<name>/skill.md`

#### Skill 文件格式

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

`frontmatter` 仅需 `name`（合法字符 `[A-Za-z0-9_-]+`）和 `description`。

保留名（不可用作 skill name）：`report` / `clear` / `help` / `quit` / `reload_skills`。

#### 激活 skill

- 用户输入 `/<skill-name>`（如 `/my-skill`）
- 或让 agent 调用 `read_file('.finagent/skills/<skill-name>/skill.md')`
- 两种方式产出等价的对话历史

#### 渐进式披露

skill.md 仅包含指令；大型资源（模板、脚本）放在 skill 目录的 `assets/` /
`scripts/` 子目录下，agent 按需通过 `read_file(path="skills/my-skill/assets/x.md")`
加载。

#### 热更新

新增/修改/删除 skill 文件后，在 TUI 中输入 `/reload_skills` 重新扫描目录。

#### 报告路径

生成的财报点评报告写入 `./.finagent/reports/`（项目本地）。

<p align="right">(<a href="#readme-top">back to top</a>)</p>
