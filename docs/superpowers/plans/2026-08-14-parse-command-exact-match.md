# parse_command Exact-Match Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore `/skill-name args` to the message path by adding a single-token guard to `parse_command`.

**Architecture:** `parse_command` matches a command or skill only when input is exactly `/name` (one whitespace-delimited token after `/`). Multiple tokens (args present) fall through to the message path. This decouples frequency recording from LLM request routing.

**Tech Stack:** Python 3.11, pytest, Textual 8.2.8

## Global Constraints

- `.venv` is the project virtualenv — run tests via `.venv/bin/python -m pytest`
- Tests run sequentially (no parallel — memory)
- No stubs
- `parse_command` builtins are case-insensitive; skills are case-sensitive (filesystem paths)
- All builtins (`/quit`, `/help`, `/clear`, `/reload_skills`) are arity-0

**Spec:** `docs/superpowers/specs/2026-08-14-parse-command-exact-match-design.md`

---

### Task 1: Add single-token guard + fix all affected tests

**Files:**
- Modify: `finagent/tui.py:52-70` (the `parse_command` function)
- Modify: `tests/test_tui.py` — 4 existing tests to update, 1 new test to add

**Interfaces:**
- Consumes: `_COMMANDS` set (line 37), `skill_names` parameter
- Produces: `parse_command(text, skill_names)` → `tuple[str, str]` — unchanged signature, corrected routing

Four existing tests assert the old behavior (skill/builtin match with args). All must be updated in the same commit as the code change to keep the suite green.

- [ ] **Step 1: Write new failing test — skill with args → message**

Add to `tests/test_tui.py` after `test_parse_command_tab_args_skill`:

```python
def test_parse_command_skill_with_args_is_message():
    """Skill name followed by args must fall through to message path."""
    sn = frozenset({"earnings-review"})
    assert parse_command("/earnings-review 000001 2025q3", skill_names=sn) == ("message", "/earnings-review 000001 2025q3")
```

- [ ] **Step 2: Write new failing test — exact skill match still works**

Add to `tests/test_tui.py`:

```python
def test_parse_command_skill_alone_no_args_matches():
    """Skill name alone (no args) still matches as skill."""
    sn = frozenset({"earnings-review"})
    assert parse_command("/earnings-review", skill_names=sn) == ("skill", "earnings-review")
```

- [ ] **Step 3: Run new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tui.py::test_parse_command_skill_with_args_is_message tests/test_tui.py::test_parse_command_skill_alone_no_args_matches -v`
Expected: `test_parse_command_skill_with_args_is_message` FAILS (returns skill, not message). `test_parse_command_skill_alone_no_args_matches` may PASS already (exact match already works).

- [ ] **Step 4: Add `len(tokens) > 1` guard to `parse_command`**

Replace the body of `parse_command` in `finagent/tui.py` (lines 52-70) with:

```python
def parse_command(text: str, skill_names: frozenset[str] = frozenset()) -> tuple[str, str]:
    """Parse user input into (command_name, payload).

    Returns ("message", text) for non-command input.
    For slash commands, returns (command_name_without_slash, "").
    For /<skill-name> matching an active skill, returns ("skill", skill_name).

    Only matches when input is exactly /name (single token, no args).
    Any args after the name cause the entire input to fall through to message.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return "message", stripped
    tokens = stripped[1:].split()
    if not tokens or len(tokens) > 1:
        return "message", stripped
    cmd = tokens[0].lower()
    if "/" + cmd in _COMMANDS:
        return cmd, ""
    if tokens[0] in skill_names:
        return "skill", tokens[0]
    return "message", stripped
```

- [ ] **Step 5: Update `test_parse_command_inline_args_skill`**

Find in `tests/test_tui.py`:

```python
def test_parse_command_inline_args_skill():
    sn = frozenset({"earnings-review"})
    assert parse_command("/earnings-review 000001", skill_names=sn) == ("skill", "earnings-review")
```

Replace with:

```python
def test_parse_command_inline_args_is_message():
    """Skill name with inline args falls through to message path."""
    sn = frozenset({"earnings-review"})
    assert parse_command("/earnings-review 000001", skill_names=sn) == ("message", "/earnings-review 000001")
```

- [ ] **Step 6: Update `test_parse_command_newline_args_skill`**

Find in `tests/test_tui.py`:

```python
def test_parse_command_newline_args_skill():
    sn = frozenset({"earnings-review"})
    assert parse_command("/earnings-review\n000001", skill_names=sn) == ("skill", "earnings-review")
```

Replace with:

```python
def test_parse_command_newline_args_is_message():
    """Skill name with newline-separated args falls through to message path."""
    sn = frozenset({"earnings-review"})
    result = parse_command("/earnings-review\n000001", skill_names=sn)
    assert result == ("message", "/earnings-review\n000001")
```

- [ ] **Step 7: Update `test_parse_command_tab_args_skill`**

Find in `tests/test_tui.py`:

```python
def test_parse_command_tab_args_skill():
    sn = frozenset({"earnings-review"})
    assert parse_command("/earnings-review\t000001", skill_names=sn) == ("skill", "earnings-review")
```

Replace with:

```python
def test_parse_command_tab_args_is_message():
    """Skill name with tab-separated args falls through to message path."""
    sn = frozenset({"earnings-review"})
    assert parse_command("/earnings-review\t000001", skill_names=sn) == ("message", "/earnings-review\t000001")
```

- [ ] **Step 8: Update `test_parse_command_case_insensitive_builtin_with_space`**

Find in `tests/test_tui.py`:

```python
def test_parse_command_case_insensitive_builtin_with_space():
    assert parse_command("/HELP extra") == ("help", "")
```

Replace with:

```python
def test_parse_command_builtin_with_args_is_message():
    """Builtin with extra args falls through to message path."""
    assert parse_command("/HELP extra") == ("message", "/HELP extra")
```

- [ ] **Step 9: Run all parse_command tests to verify green**

Run: `.venv/bin/python -m pytest tests/test_tui.py -k "parse_command" -v`
Expected: ALL PASS

- [ ] **Step 10: Commit**

```bash
git add finagent/tui.py tests/test_tui.py
git commit -m "fix: parse_command single-token guard — /skill args falls to message"
```

---

### Task 2: Add `.finagent.json` to `.gitignore`

**Files:**
- Modify: `.gitignore` (project root)

- [ ] **Step 1: Add `.finagent.json` line**

In `.gitignore`, find the section:

```
# FinAgent runtime data (sessions, memory)
.finagent/
.superpowers/
```

Insert `.finagent.json` between `.finagent/` and `.superpowers/`:

```
# FinAgent runtime data (sessions, memory)
.finagent/
.finagent.json
.superpowers/
```

- [ ] **Step 2: Verify `.finagent.json` is now ignored**

Run: `git check-ignore .finagent.json`
Expected: prints `.finagent.json`

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore .finagent.json (local usage data)"
```

---

### Task 3: Full test suite verification

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/`
Expected: ALL PASS (277+ tests)

- [ ] **Step 2: Verify skill activation flow intact**

Run: `.venv/bin/python -m pytest tests/test_tui.py -k "skill" -v`
Expected: ALL PASS — exact-match `/skill-name` activation still works
