# parse_command exact-match fix

## Problem

`parse_command` was rewritten to extract the first whitespace-delimited token after `/` for command/skill detection. This made `/skill-name args` match the skill, routing it to `_activate_skill` which discards the args. The agent never receives the ticker/args, so analysis doesn't continue.

Before the rewrite, `/skill-name args` fell through to the message path because `stripped[1:]` (entire text after `/`) didn't exactly match any skill name. The agent received the full text, saw the skill catalog in the system-reminder, loaded the skill itself, and processed the args. This was the intended flow.

The rewrite was motivated by frequency recording: multi-line input `/skill-name\nargs` wasn't detected as a skill, so freq wasn't recorded. The fix over-corrected by matching the skill name even when args were present.

## Design decision

**Single-token exact match only.** `parse_command` matches a command or skill only when the input is exactly `/name` — one token after `/`, no args. Any additional tokens (args, ticker codes) cause the entire input to fall through to the message path.

This decouples frequency recording from LLM request routing:
- `/earnings-review` (exact) → skill activation, freq +1
- `/earnings-review 000001 2025q3` → message path, agent handles everything, freq not recorded
- `/help` (exact) → builtin command, freq +1
- `/help extra` → message path, freq not recorded

## Implementation

### parse_command change

Add a `len(tokens) == 1` guard before matching:

```python
def parse_command(text: str, skill_names: frozenset[str] = frozenset()) -> tuple[str, str]:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return "message", stripped
    tokens = stripped[1:].split()
    if not tokens:
        return "message", stripped
    if len(tokens) > 1:
        return "message", stripped
    cmd = tokens[0].lower()
    if "/" + cmd in _COMMANDS:
        return cmd, ""
    if tokens[0] in skill_names:
        return "skill", tokens[0]
    return "message", stripped
```

`str.split()` normalizes all whitespace (trailing spaces, tabs, newlines) so `/help ` and `/help` both produce `["help"]` — single token, matches.

### .gitignore

Add `.finagent.json` to `.gitignore`. The file is local usage data, not project configuration.

### Frequency recording

No change. `on_chat_input_submitted` already records freq only when `parse_command` returns a command/skill. With exact-match, freq is recorded only for standalone commands/skills — the intended behavior.

## Test changes

Two existing edge-case tests assert behavior that contradicts this design:

- `test_parse_command_inline_args_skill`: `/earnings-review 000001` was expected to return `("skill", ...)` — now must return `("message", ...)`.
- `test_parse_command_case_insensitive_builtin_with_space`: `/HELP extra` was expected to return `("help", "")` — now must return `("message", ...)`.

Update both to assert message-path behavior.

Add one new test:
- `test_parse_command_skill_alone_no_args_matches`: `/earnings-review` with no args → `("skill", "earnings-review")`. Confirms exact match still works.

## What stays unchanged

- Popup trigger/matching/sorting logic
- Tab/Enter/Up/Down popup behavior
- Frequency persistence (`CommandFreq` module)
- `_activate_skill` behavior
- Agent system-reminder skill catalog injection

## Notes

- **Case sensitivity asymmetry**: builtins match case-insensitively (`/HELP` → `help`); skills match case-sensitively (`/Earnings-Review` ≠ `earnings-review`). This is intentional — skill names are filesystem paths (`skills/<name>/skill.md`), which are case-sensitive on macOS/Linux.
- **Builtin arity-0 contract**: all builtins (`/quit`, `/help`, `/clear`, `/reload_skills`) take no arguments. If a future builtin needs args, it must be restructured as a skill (args go through the message path, agent handles them).
