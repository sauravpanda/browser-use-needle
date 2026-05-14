# browser-use-needle

Small local experiment for trying [cactus-compute/needle](https://github.com/cactus-compute/needle) with the sibling Browser Use repos.

This repo does not modify `../browser-use` or `../browser-use-rust`. It installs `../browser-use` editable and Needle from GitHub, then exposes a `buneedle` CLI for generating tool calls.

## Setup

```bash
uv sync --python 3.11
uv run buneedle doctor
```

The first real Needle run downloads `Cactus-Compute/needle/needle.pkl` and the tokenizer from Hugging Face into local caches/checkpoints, then JAX compiles the model.

## Try the Rust Tool-Call Preset

`../browser-use-rust` is a useful first target because the action space is small: click, type, scroll, hotkey, done.

```bash
uv run buneedle tools --preset browser-use-rust --pretty

uv run buneedle run \
  --preset browser-use-rust \
  --query "The screen shows a search page. Click the search input near the middle of the page." \
  --rust-action
```

`--rust-action` converts Needle's JSON tool call into the current Rust parser format, for example:

```text
ACTION: click 512 384
```

Needle is text-only, so it cannot directly replace the vision step in the Rust screenshot loop. It can test the tool-selection/tool-argument part once you have a text state description from a vision model, DOM state, accessibility tree, or another screen parser.

## Try the Python Browser Use Adapter

```bash
uv run buneedle tools --preset browser-use-python-core --pretty

uv run python examples/browser_use_agent.py \
  --task "Go to https://example.com and report the page heading" \
  --max-steps 4
```

The `ChatNeedle` adapter implements Browser Use's `BaseChatModel` protocol and maps Needle tool calls back into Browser Use `AgentOutput` actions. By default it exposes only a core action subset (`done`, `search`, `navigate`, `go_back`, `wait`, `click`, `input`, `scroll`, `send_keys`) because the full Browser Use action space is noisy for a 26M model. This is experimental: Browser Use normally expects a stronger reasoning model and may include visual content that Needle does not use. Start with simple text/DOM-driven tasks and `use_vision=False`.

Current smoke-test behavior:

- Single-step tool-call generation works for small prompts, e.g. `browser-use-rust` returns `ACTION: type hello world`.
- The Browser Use adapter can validate simple actions, e.g. `navigate` for `https://example.com`.
- A multi-step Browser Use task against Hacker News navigated successfully, then Needle emitted invalid JSON from the longer page-state prompt. The adapter now converts that into a clean failed `done` action instead of retrying the same malformed output.

## Useful Commands

```bash
# Print the preset tool schema Needle receives.
uv run buneedle tools --preset browser-use-rust --pretty

# Run a single tool-call generation and print JSON.
uv run buneedle run --preset browser-use-rust --query "Type hello into the focused field"

# Read a query from stdin.
printf '%s\n' "Scroll down one page" | uv run buneedle run --preset browser-use-rust --rust-action

# Launch Needle's own playground UI.
uv run buneedle playground --port 7860
```
