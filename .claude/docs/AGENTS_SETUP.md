# Subagent Setup — Opus orchestrates, Sonnet implements

How to make Claude Code run its main session on Opus while all coding is done by Sonnet subagents.

---

## The model

- **Opus (main session)** — architect. Decides design, writes specs, reads results, reviews output, explains things to you. Never types out modules itself.
- **Sonnet (subagents)** — implementers. Each gets a precise spec, works in its own context window, returns a summary.

Two things you gain: Opus reasoning stays focused on decisions instead of being spent on boilerplate, and verbose output (test logs, file contents, search results) stays inside subagent contexts rather than filling your main conversation.

---

## Files to install

Copy the `.claude/` folder to your repository root:

```
neurovision-x/
├── CLAUDE.md
└── .claude/
    ├── settings.json
    └── agents/
        ├── py-implementer.md
        ├── test-runner.md
        ├── code-reviewer.md
        └── docs-writer.md
```

`.claude/` starts with a dot, so it's hidden in Finder — press <kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>.</kbd> to see it. Commit the whole folder to git; these definitions are part of the project.

If you'd rather not move files by hand, paste this into Claude Code instead:

```
Create the following in this repository:

.claude/settings.json with an "env" block setting CLAUDE_CODE_SUBAGENT_MODEL to "sonnet".

.claude/agents/py-implementer.md — model sonnet, tools Read/Write/Edit/Bash/Grep/Glob.
Implements one Python module at a time from a precise spec, plus its pytest tests. Runs
pytest and ruff before returning. Forbidden from: adding unrequested features, adding
dependencies, hardcoding paths, writing "cuda" anywhere, writing tests needing a GPU or
real data, or touching files outside the spec.

.claude/agents/test-runner.md — model sonnet, tools Read/Bash/Grep/Glob. Runs pytest,
smoke tests, and lint. Reports only pass/fail counts plus, per failure, the test name,
assertion, location, and a one-sentence diagnosis. Never fixes code, never edits files,
never pastes raw logs.

.claude/agents/code-reviewer.md — model sonnet, tools Read/Grep/Glob/Bash, read-only.
Reviews a module against its spec and the hard constraints in CLAUDE.md: hardcoded
paths, device assumptions, 16GB VRAM violations, incomplete checkpoint state, new
dependencies, missing shape tests, plus shape/dtype/label-mapping correctness. Outputs
findings grouped as Critical / Should fix / Consider.

.claude/agents/docs-writer.md — model sonnet, tools Read/Write/Edit/Grep/Glob. Writes
docstrings, MkDocs pages, README sections, and appends append-only rows to
docs/experiments.md. Never modifies code, never invents numbers.

Then restart yourself so the new agents directory is picked up.
```

---

## Two ways the model gets set, and why both

**1. `model: sonnet` in each agent's frontmatter.** The documented per-agent setting.

**2. `CLAUDE_CODE_SUBAGENT_MODEL: "sonnet"` in `.claude/settings.json`.** The environment variable sits at the top of the resolution order — above the per-invocation parameter and above frontmatter — so it forces every subagent onto Sonnet regardless of what anything else says.

Belt and braces. There have been reports of frontmatter `model` being ignored in some versions, and the env var is the setting that cannot be missed.

**The trade-off:** because the env var wins over everything, you can no longer route an individual agent to a cheaper model. If you later want a Haiku agent for bulk search, delete the env var from `settings.json` and rely on frontmatter alone.

---

## Starting a session

```bash
cd neurovision-x
claude --model opus
```

Then confirm it took:

```
/model
```

should report Opus for the main conversation. To verify delegation works:

```
Use the test-runner subagent to run the test suite and report the result.
```

Check the task panel below the prompt — the subagent row should show Sonnet. If Claude Code says it can't find the agent, restart it: a running session doesn't detect a `.claude/agents/` directory that didn't exist when it started.

---

## Invoking agents

| How | When |
|---|---|
| Say nothing | Claude delegates on its own based on each agent's `description` |
| Name it in prose — *"have py-implementer build this"* | You want it delegated but trust Claude's framing |
| `@agent-py-implementer` | You want to guarantee that specific agent runs |

Your full message always goes to Opus, which writes the subagent's task prompt. The @-mention picks *which* agent, not *what* it's told — so you still describe the task to Opus in your own words.

---

## What must stay on Opus

Delegation is a cost optimization, not a division of intellectual labor. Keep these in the main session:

- Architectural and research decisions
- Interpreting loss curves, metrics, ablation tables, failure cases
- Writing the specs the subagents implement
- Statistical claims and anything that goes in the paper
- **Explaining delegated code back to you** — you're learning this material, and code you don't understand is a liability regardless of which model wrote it

If Opus starts writing modules directly, remind it: *"delegate that to py-implementer with a spec."*

---

## Things worth knowing

**Subagents load CLAUDE.md.** Every custom subagent gets the full CLAUDE.md hierarchy, so your hard constraints reach them automatically. The built-in Explore and Plan agents are the exception — they skip it for speed.

**Subagents don't see your conversation.** Each starts with a fresh context containing only its system prompt, CLAUDE.md, and the task message Opus writes. Anything from your discussion that matters must be restated in the spec. This is the single most common reason a delegation comes back wrong.

**Subagents run in the background by default.** They work while you keep talking to Opus. Background agents have a reduced tool set, though the ones defined here stay within it. Press <kbd>Ctrl</kbd>+<kbd>B</kbd> to background a foreground task, and `/tasks` to see what's running.

**You can resume a subagent.** Ask Opus to continue a previous agent's work and it keeps its full history rather than starting over — useful when a module needs a second pass.

**Nesting is allowed.** A subagent can spawn its own, a few layers deep. You won't need this; if implementation starts spawning implementation, the spec was too big.

---

## The loop in practice

```
You:    "Implement the checkpoint module."

Opus:   Decides what state must be saved and why, states the trade-off on
        file size vs. completeness, writes a spec: file path, function
        signatures, the exact keys in the checkpoint dict, the resume test.

Sonnet: py-implementer writes training/checkpoint.py and tests/test_checkpoint.py,
        runs pytest and ruff, returns a summary.

Sonnet: code-reviewer reads it against the spec, flags that the RNG state
        for numpy isn't restored.

Opus:   Re-delegates the fix, then explains to you what a checkpoint actually
        contains and why a half-restored one silently corrupts a training run.
```

Your job in that loop is the first line and the last — deciding what to build, and understanding what came back.
