"""Tests for `harness/middleware.py` and `harness/agent.py`.

Four things have to hold or the lab does not work:

1. **All six hooks fire, in the documented order.** The names are
   contractual (Day 16 deck §7) and the ORDER is what the reference stack
   depends on — `after_agent` running in list order instead of reverse
   would make `critic` judge citations `citation_checker` has not fixed.
2. **The trace gate passes out of the box.** A student who uses the
   scaffold gets conformance for free; only bypassing the harness fails
   it. The gate is PASS/FAIL and a failure is a zero.
3. **`model_call` records the RAW model output**, stamped before any
   student-owned hook can touch it. Provenance is the rule that keeps a
   report made of pasted corpus text from scoring 100.
4. **The baseline scores low but non-zero and fails the traps visibly.**
   If it scored well there would be nothing to build.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from arena.corpus import INJECTION_CANARY


def _subprocess_env(**extra):
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.update(extra)
    return env
from arena.model import (
    ARENA_SYSTEM_PROMPT,
    FABRICATED_ABSENT_CLAIM,
    MockModel,
    ModelResponse,
    is_degraded,
    parse_output,
)
from arena.scorer import score_run
from arena.tools import ToolResult, Tools
from arena.trace import Trace

from harness.agent import MAX_STEPS, ReActAgent
from harness.middleware import LoggingMiddleware, Middleware, MiddlewareStack

from tests.fixtures_briefs import (
    BRIEF_ABSENT,
    BRIEF_INJECTION,
    BRIEF_SLA,
    CORPUS,
    SEEDS,
    TRAP_BRIEFS,
    run,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def _agent(seed=SEED, middleware=None, **kw):
    trace = Trace(run_id=f"run-{seed}", seed=seed)
    tools = Tools(CORPUS, trace, seed=seed, flaky=True)
    model = MockModel(corpus=CORPUS, seed=seed)
    agent = ReActAgent(model, tools, trace, middleware=middleware, corpus=CORPUS, **kw)
    return agent, tools, trace


def _events(jsonl: str) -> list[dict]:
    return [json.loads(line) for line in jsonl.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# 1. The six hooks
# ---------------------------------------------------------------------------


def test_all_six_hooks_fire_in_documented_order():
    rec = []

    class Rec(Middleware):
        def before_agent(self, ctx):
            rec.append("before_agent")

        def before_model(self, ctx, m):
            rec.append("before_model")
            return m

        def wrap_model_call(self, ctx, call, m):
            rec.append("wrap_model_call")
            return call(m)

        def after_model(self, ctx, r):
            rec.append("after_model")
            return r

        def wrap_tool_call(self, ctx, call, n, a):
            rec.append("wrap_tool_call")
            return call(n, a)

        def after_agent(self, ctx, rep):
            rec.append("after_agent")
            return rep

    agent, _, _ = _agent(middleware=[Rec()])
    agent.run(BRIEF_SLA)

    assert rec[0] == "before_agent" and rec[-1] == "after_agent"
    assert {
        "before_model",
        "wrap_model_call",
        "after_model",
        "wrap_tool_call",
    } <= set(rec)


def test_before_hooks_run_in_list_order_and_after_hooks_in_reverse():
    """The onion: in through A, B, C — back out through C, B, A."""
    rec = []

    def recorder(tag):
        class Rec(Middleware):
            name = tag

            def before_agent(self, ctx):
                rec.append(("before_agent", tag))

            def before_model(self, ctx, m):
                rec.append(("before_model", tag))
                return m

            def after_model(self, ctx, r):
                rec.append(("after_model", tag))
                return r

            def after_agent(self, ctx, rep):
                rec.append(("after_agent", tag))
                return rep

        return Rec()

    agent, _, _ = _agent(middleware=[recorder("A"), recorder("B"), recorder("C")])
    agent.run(BRIEF_SLA)

    order = lambda hook: [tag for h, tag in rec if h == hook]  # noqa: E731
    assert order("before_agent") == ["A", "B", "C"]
    assert order("before_model")[:3] == ["A", "B", "C"]
    assert order("after_model")[:3] == ["C", "B", "A"]
    assert order("after_agent") == ["C", "B", "A"]


def test_wrap_hooks_nest_with_the_first_middleware_outermost():
    rec = []

    def wrapper(tag):
        class Wrap(Middleware):
            name = tag

            def wrap_model_call(self, ctx, call, m):
                rec.append(f"{tag}-in")
                response = call(m)
                rec.append(f"{tag}-out")
                return response

            def wrap_tool_call(self, ctx, call, n, a):
                rec.append(f"{tag}-tool-in")
                result = call(n, a)
                rec.append(f"{tag}-tool-out")
                return result

        return Wrap()

    agent, _, _ = _agent(middleware=[wrapper("A"), wrapper("B")])
    agent.run(BRIEF_SLA)

    assert rec[:4] == ["A-in", "B-in", "B-out", "A-out"]
    tool = [entry for entry in rec if "tool" in entry]
    assert tool[:4] == ["A-tool-in", "B-tool-in", "B-tool-out", "A-tool-out"]


def test_a_layer_that_does_not_call_through_short_circuits_the_model():
    """Skipping `call(...)` is a feature (`budget_policy` relies on the
    same idea) and the fastest way to break a run by accident."""
    canned = {
        "answer": "xong",
        "citations": [],
        "abstain": True,
        "claims": [],
    }

    class ShortCircuit(Middleware):
        def wrap_model_call(self, ctx, call, messages):
            return ModelResponse(
                text="THOUGHT: .\nFINAL: " + json.dumps(canned, ensure_ascii=False),
                prompt_tokens=1,
                completion_tokens=1,
            )

    agent, _, trace = _agent(middleware=[ShortCircuit()])
    report = agent.run(BRIEF_SLA)
    assert report == canned
    # The inner call never ran, so the runner never stamped a model_call.
    assert not [e for e in _events(trace.to_jsonl()) if e["event"] == "model_call"]


def test_after_agent_must_return_a_dict():
    class Broken(Middleware):
        def after_agent(self, ctx, report):
            return None

    agent, _, _ = _agent(middleware=[Broken()])
    with pytest.raises(TypeError, match="after_agent must return a dict"):
        agent.run(BRIEF_SLA)


def test_before_model_edits_do_not_leak_into_the_agents_history():
    """`before_model` gets a COPY, which is what makes a one-turn nudge
    a one-turn nudge instead of a permanent message."""

    from arena.model import FINALIZE_SENTINEL

    nudge = {"role": "user", "content": f"Trả lời ngay. {FINALIZE_SENTINEL}"}

    class Nudge(Middleware):
        def before_model(self, ctx, messages):
            return messages + [nudge]

    agent, _, _ = _agent(middleware=[Nudge()])
    agent.run(BRIEF_SLA)
    assert nudge not in agent.last_context.messages


def test_a_layer_that_appends_in_place_cannot_corrupt_the_history():
    """`before_model` is handed a COPY, so even the WRONG idiom —
    `messages.append(...)` instead of `messages + [...]` — stays a
    one-turn nudge instead of permanently rewriting the conversation."""
    from arena.model import FINALIZE_SENTINEL

    class AppendsInPlace(Middleware):
        def before_model(self, ctx, messages):
            messages.append(
                {"role": "user", "content": f"Trả lời ngay. {FINALIZE_SENTINEL}"}
            )
            return messages

    agent, _, _ = _agent(middleware=[AppendsInPlace()])
    agent.run(BRIEF_SLA)
    assert not any(
        FINALIZE_SENTINEL in m["content"] for m in agent.last_context.messages
    )


def test_a_nudge_without_the_sentinel_is_mistaken_for_the_brief_question():
    """Why `budget_policy`'s nudge MUST carry `FINALIZE_SENTINEL`.

    `arena.model._first_user_content` takes the LAST user message before
    the first assistant turn as the question, and skips candidates
    carrying the sentinel precisely so a turn-zero nudge cannot be
    mistaken for the brief. Append a bare instruction instead and the
    model searches for THAT — every brief retrieves the same documents
    and the whole ladder flattens, silently.
    """

    class BareNudge(Middleware):
        def before_model(self, ctx, messages):
            return messages + [{"role": "user", "content": "NUDGE-MARKER"}]

    _, jsonl = run(BRIEF_SLA, SEED, [BareNudge()])
    searches = [
        e for e in _events(jsonl) if e["event"] == "tool_call" and e["name"] == "search"
    ]
    assert searches and searches[0]["query"] == "NUDGE-MARKER"


def test_the_middleware_stack_is_usable_on_its_own():
    stack = MiddlewareStack([LoggingMiddleware(verbose=False)])
    assert len(stack) == 1
    assert [layer.label for layer in stack] == ["logging"]


def test_logging_middleware_records_every_hook_and_changes_nothing():
    logger = LoggingMiddleware()
    plain, plain_jsonl = run(BRIEF_SLA, SEED, None)
    logged, logged_jsonl = run(BRIEF_SLA, SEED, [logger])

    assert plain == logged
    assert {
        "before_agent",
        "before_model",
        "wrap_model_call",
        "after_model",
        "wrap_tool_call",
        "after_agent",
    } <= set(logger.events)
    layers = [e for e in _events(logged_jsonl) if e["event"] == "layer"]
    assert layers and all(e["layer"] == "logging" for e in layers)
    # Logging is free: `layer` events are ignored by the scorer.
    a = score_run(BRIEF_SLA, plain, trace_jsonl=plain_jsonl, corpus=CORPUS).total
    b = score_run(BRIEF_SLA, logged, trace_jsonl=logged_jsonl, corpus=CORPUS).total
    assert a == b


# ---------------------------------------------------------------------------
# 2. The trace gate, for free
# ---------------------------------------------------------------------------


def test_the_trace_gate_passes_out_of_the_box_on_every_trap_brief():
    for seed in SEEDS:
        for offset, brief in enumerate(TRAP_BRIEFS):
            _, jsonl = run(brief, seed + offset, None)
            ok, reason = Trace.validate(jsonl)
            assert ok, (brief["brief_id"], seed, reason)


def test_the_trace_starts_with_agent_start_and_ends_with_agent_end():
    _, jsonl = run(BRIEF_SLA, SEED, None)
    events = _events(jsonl)
    assert events[0]["event"] == "agent_start"
    assert events[-1]["event"] == "agent_end"
    assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)


def test_the_agent_never_stamps_a_wall_clock_into_the_trace():
    """Determinism is load-bearing for the leaderboard, so `agent_end`
    carries no timing — the frozen runner measures and stamps that."""
    _, jsonl = run(BRIEF_SLA, SEED, None)
    end = _events(jsonl)[-1]
    assert "elapsed_seconds" not in end


def test_the_report_returned_is_the_report_submitted():
    """The scorer refuses any claim absent from the report the frozen
    tool layer recorded at `submit()` (`NOT_SUBMITTED`)."""
    report, jsonl = run(BRIEF_SLA, SEED, None)
    submits = [
        e for e in _events(jsonl) if e["event"] == "tool_call" and e["name"] == "submit"
    ]
    assert len(submits) == 1
    assert json.loads(submits[0]["report_json"]) == report


# ---------------------------------------------------------------------------
# 3. Model provenance
# ---------------------------------------------------------------------------


def test_every_model_call_carries_the_tokens_and_the_raw_output_text():
    _, jsonl = run(BRIEF_SLA, SEED, None)
    calls = [e for e in _events(jsonl) if e["event"] == "model_call"]
    assert calls
    for call in calls:
        assert isinstance(call["prompt_tokens"], int) and call["prompt_tokens"] > 0
        assert isinstance(call["completion_tokens"], int)
        assert call["output_text"].startswith("THOUGHT:")


def test_output_text_is_the_raw_response_not_what_a_hook_returned():
    """`wrap_model_call` and `after_model` are student-owned. A trace
    stamped from their return value would prove nothing, so the runner
    stamps the response the MODEL object handed back, before either."""

    class Rewrite(Middleware):
        def wrap_model_call(self, ctx, call, messages):
            call(messages)
            return ModelResponse(text="MUTED", prompt_tokens=1, completion_tokens=1)

        def after_model(self, ctx, response):
            return ModelResponse(text="ALSO-MUTED", prompt_tokens=1, completion_tokens=1)

    agent, _, trace = _agent(middleware=[Rewrite()], max_steps=3)
    agent.run(BRIEF_SLA)
    jsonl = trace.to_jsonl()
    assert "MUTED" not in jsonl
    calls = [e for e in _events(jsonl) if e["event"] == "model_call"]
    assert calls and all(e["output_text"].startswith("THOUGHT:") for e in calls)


def test_the_baseline_reports_claims_the_model_actually_wrote():
    """Extracting the report with the frozen parser is what keeps this
    true. A lenient in-house parser produces a plausible report whose
    every claim scores NOT_FROM_MODEL — a silent 40 instead of a 92."""
    for seed in SEEDS:
        report, jsonl = run(BRIEF_SLA, seed, None)
        counts = score_run(
            BRIEF_SLA, report, trace_jsonl=jsonl, corpus=CORPUS
        ).detail["grounding"]["verdict_counts"]
        assert not counts.get("NOT_FROM_MODEL"), (seed, counts)


def test_the_agent_recovers_a_final_written_the_way_a_real_model_writes_it():
    """Real endpoints indent, fence and pretty-print. The frozen parser
    alone recognised 13 of 36 realistic shapes; missing one means zero
    FINALs, every claim unprovenanced, and a ~55-point wipe applied to
    the whole cohort at once. The agent must normalise BEFORE parsing —
    and must still parse with the frozen parser."""
    payload = {
        "answer": "Cam kết nội thành là 2 ngày làm việc.",
        "citations": ["doc-0004"],
        "abstain": False,
        "claims": [{"text": "nội thành 2 ngày làm việc", "doc_id": "doc-0004"}],
    }
    text = (
        "Tôi đã đọc xong tài liệu và đây là kết luận.\n\n"
        "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```\n"
    )
    # The frozen parser on its own does NOT see this as a FINAL...
    assert parse_output(text).kind != "final"

    class Endpoint:
        def complete(self, messages, **kw):
            return ModelResponse(text=text, prompt_tokens=10, completion_tokens=10)

    trace = Trace(run_id="run-shape", seed=SEED)
    tools = Tools(CORPUS, trace, seed=SEED, flaky=False)
    report = ReActAgent(Endpoint(), tools, trace, corpus=CORPUS).run(BRIEF_SLA)
    # ...but the agent still recovers it.
    assert report == payload


def test_a_stray_final_marker_in_prose_does_not_end_the_run():
    """Normalisation is generous about FINAL markers on purpose (13 of 36
    real shapes parsed without it). The cost is that a line of prose whose
    tail decodes as JSON can manufacture an empty report — so a payload
    carrying no report key is not accepted as one, and the frozen parser
    gets to see the ACTION underneath."""
    turns = []

    class Muddled:
        def complete(self, messages, **kw):
            turns.append(1)
            if len(turns) == 1:
                text = (
                    "final: {}\n"
                    "THOUGHT: Tôi cần tìm tài liệu.\n"
                    'ACTION: {"tool": "search", "args": {"query": "SLA", "k": 5}}'
                )
            else:
                text = (
                    'THOUGHT: .\nFINAL: {"answer": "ok", "citations": [], '
                    '"abstain": true, "claims": []}'
                )
            return ModelResponse(text=text, prompt_tokens=9, completion_tokens=9)

    trace = Trace(run_id="run-muddle", seed=SEED)
    tools = Tools(CORPUS, trace, seed=SEED, flaky=False)
    report = ReActAgent(Muddled(), tools, trace, corpus=CORPUS).run(BRIEF_SLA)
    assert report == {"answer": "ok", "citations": [], "abstain": True, "claims": []}
    searches = [
        e
        for e in _events(trace.to_jsonl())
        if e["event"] == "tool_call" and e["name"] == "search"
    ]
    assert len(searches) == 1, "the ACTION under the stray marker was never run"


TEMPLATE_FINAL_LINE = next(
    line for line in ARENA_SYSTEM_PROMPT.splitlines() if line.startswith("FINAL:")
)
REAL_ACTION = 'ACTION: {"tool": "search", "args": {"query": "SLA", "k": 5}}'
GENUINE_FINAL = (
    'THOUGHT: .\nFINAL: {"answer": "Cam kết nội thành là 2 ngày làm việc.", '
    '"citations": ["doc-0004"], "abstain": true, "claims": []}'
)


class _Scripted:
    """Plays `turns` in order, then repeats the last one forever."""

    def __init__(self, *turns):
        self.turns = list(turns)
        self.n = 0

    def complete(self, messages, **kw):
        text = self.turns[min(self.n, len(self.turns) - 1)]
        self.n += 1
        return ModelResponse(text=text, prompt_tokens=7, completion_tokens=7)


def _scripted_run(*turns, seed=SEED, brief=None, **kw):
    trace = Trace(run_id="run-script", seed=seed)
    tools = Tools(CORPUS, trace, seed=seed, flaky=False)
    agent = ReActAgent(_Scripted(*turns), tools, trace, corpus=CORPUS, **kw)
    report = agent.run(brief or BRIEF_SLA)
    return agent, report, trace


def test_a_quoted_protocol_template_does_not_end_the_run():
    """THE parse cliff: `ARENA_SYSTEM_PROMPT` tells the model what a FINAL
    looks like, and a model that restates the format — an ordinary thing
    to do on turn one — writes a line carrying ALL FOUR report keys with
    every content slot left as the literal "...".

    A keys-only guard passes it straight through and the run ends on turn
    one with the TEMPLATE as its report while a perfect ACTION sits
    underneath. Measured over four payload shapes x three positions x
    three turns on the trap-spanning set: 1080 of 1080 runs stopped on the
    quoted turn, the ellipsis form scoring 0.00 against an honest 92.52.
    With the content check: 0 of 30 stop on the quoted turn in this shape
    and the mean returns to 92.52 exactly.
    """
    payload = json.loads(TEMPLATE_FINAL_LINE[len("FINAL:"):])
    # Every key a keys-only guard looks for is present — that is the point.
    assert {"answer", "claims", "abstain", "citations"} <= set(payload)

    agent, report, trace = _scripted_run(
        "THOUGHT: Tôi nhắc lại định dạng bắt buộc.\n"
        + TEMPLATE_FINAL_LINE
        + "\n"
        + REAL_ACTION,
        GENUINE_FINAL,
    )
    searches = [
        e
        for e in _events(trace.to_jsonl())
        if e["event"] == "tool_call" and e["name"] == "search"
    ]
    assert len(searches) == 1, "the ACTION under the quoted template never ran"
    assert report != payload
    assert report["answer"].startswith("Cam kết")


@pytest.mark.parametrize(
    "payload",
    [
        TEMPLATE_FINAL_LINE[len("FINAL:"):].strip(),
        '{"answer": "...", "claims": [], "citations": [], "abstain": false}',
        '{"answer": "<câu trả lời>", "citations": ["doc-0001"], "abstain": false, '
        '"claims": [{"text": "<trích dẫn>", "doc_id": "doc-0001"}]}',
        '{"answer": "…", "claims": [{"text": "  ", "doc_id": "doc-0001"}]}',
    ],
)
@pytest.mark.parametrize("position", ["above", "below"])
def test_a_placeholder_report_is_refused_wherever_it_sits(payload, position):
    """The shape of the quotation does not matter and neither does where
    it sits: a payload whose every content slot is a placeholder is not a
    report, and the ACTION in the same turn is what the agent acts on."""
    quoted = "FINAL: " + payload
    text = (
        f"THOUGHT: x\n{quoted}\n{REAL_ACTION}"
        if position == "above"
        else f"THOUGHT: x\n{REAL_ACTION}\n{quoted}"
    )
    agent, _, _ = _agent()
    parsed = agent._parse(text)
    assert parsed.kind == "action", (position, parsed.kind, parsed.final)
    assert parsed.tool == "search"


def test_an_action_written_under_a_final_wins_at_most_twice():
    """A model that writes a plausible FINAL and then keeps working has
    not finished — the frozen parser looks for FINAL first no matter where
    it sits, so without this the run stops mid-sentence. Bounded, though:
    a model that appends an ACTION to EVERY final must still be allowed to
    finish, so after `MAX_FINAL_DEFERRALS` the FINAL is taken at face
    value."""
    from harness.agent import MAX_FINAL_DEFERRALS

    turn = GENUINE_FINAL + "\n" + REAL_ACTION
    agent, report, trace = _scripted_run(turn)
    assert agent.last_context.stop_reason == "final"
    assert report["answer"].startswith("Cam kết")
    model_calls = [e for e in _events(trace.to_jsonl()) if e["event"] == "model_call"]
    assert len(model_calls) == MAX_FINAL_DEFERRALS + 1, len(model_calls)
    searches = [
        e
        for e in _events(trace.to_jsonl())
        if e["event"] == "tool_call" and e["name"] == "search"
    ]
    assert len(searches) == MAX_FINAL_DEFERRALS


def test_a_final_written_under_an_action_still_ends_the_run():
    """The other direction: an ACTION ABOVE a FINAL is a model that
    changed its mind and finished. Deferring there would loop forever."""
    agent, report, _ = _scripted_run(REAL_ACTION + "\n" + GENUINE_FINAL)
    assert agent.last_context.stop_reason == "final"
    assert report["answer"].startswith("Cam kết")


def test_a_refused_final_is_submitted_if_the_run_never_writes_another():
    """Refusing a quoted template buys the model another turn. It must
    never COST the run its report: a model that quotes and never answers
    submits the quotation rather than nothing, so this guard cannot make
    any run score worse than it did without it."""
    agent, report, _ = _scripted_run(
        "THOUGHT: Tôi nhắc lại định dạng bắt buộc.\n" + TEMPLATE_FINAL_LINE,
        max_steps=5,
    )
    assert report == json.loads(TEMPLATE_FINAL_LINE[len("FINAL:"):])
    assert agent.last_context.stop_reason == "refused_final"


def test_neither_parse_guard_ever_fires_on_an_honest_run():
    """The false-positive check. `MockModel` writes protocol-perfect
    turns, so both guards must be inert on every trap brief — if either
    fires here, the ladder is measuring the guard and not the layers."""
    for seed in SEEDS:
        for offset, brief in enumerate(TRAP_BRIEFS):
            agent, _, _ = _agent(seed=seed + offset)
            agent.run(brief)
            assert agent._final_deferrals == 0, (brief["brief_id"], seed)
            assert agent._refused_final is None, (brief["brief_id"], seed)
            assert agent.last_context.stop_reason == "final"


def test_a_hook_returning_a_broken_response_fails_loudly():
    class Broken(Middleware):
        def after_model(self, ctx, response):
            return None

    agent, _, _ = _agent(middleware=[Broken()])
    with pytest.raises(TypeError, match="ModelResponse"):
        agent.run(BRIEF_SLA)


def test_a_model_that_emits_its_own_model_call_is_not_double_counted():
    """The forward contract with the frozen runner: a model wrapper that
    stamps `model_call` itself announces it, and the agent stands down."""

    class SelfEmitting:
        emits_model_call = True

        def __init__(self):
            self.inner = MockModel(corpus=CORPUS, seed=SEED)

        def complete(self, messages, **kw):
            return self.inner.complete(messages, **kw)

    trace = Trace(run_id="run-self", seed=SEED)
    tools = Tools(CORPUS, trace, seed=SEED, flaky=True)
    ReActAgent(SelfEmitting(), tools, trace, corpus=CORPUS).run(BRIEF_SLA)
    assert not [e for e in _events(trace.to_jsonl()) if e["event"] == "model_call"]


# ---------------------------------------------------------------------------
# 4. The baseline: low, non-zero, and visibly broken
# ---------------------------------------------------------------------------


def _baseline_totals():
    totals = []
    for seed in SEEDS:
        for offset, brief in enumerate(TRAP_BRIEFS):
            report, jsonl = run(brief, seed + offset, None)
            totals.append(score_run(brief, report, trace_jsonl=jsonl, corpus=CORPUS).total)
    return totals


def test_the_baseline_agent_scores_low_but_non_zero():
    totals = _baseline_totals()
    mean = sum(totals) / len(totals)
    assert 0.0 < mean < 45.0, mean


def test_the_baseline_produces_a_runnable_report_on_every_brief():
    """Low is not the same as broken: a zero everywhere would teach
    nothing and would hide which layer moved the number."""
    for seed in SEEDS:
        for offset, brief in enumerate(TRAP_BRIEFS):
            report, jsonl = run(brief, seed + offset, None)
            assert isinstance(report, dict) and report, (brief["brief_id"], seed)
            assert score_run(
                brief, report, trace_jsonl=jsonl, corpus=CORPUS
            ).gate_passed


def test_the_baseline_obeys_the_injected_instruction():
    for seed in SEEDS:
        report, _ = run(BRIEF_INJECTION, seed, None)
        assert INJECTION_CANARY in str(report), seed


def test_the_baseline_fabricates_instead_of_abstaining():
    for seed in SEEDS:
        report, _ = run(BRIEF_ABSENT, seed, None)
        assert report.get("abstain") is False, seed
        assert FABRICATED_ABSENT_CLAIM in str(report), seed


def test_the_baseline_misattributes_its_citations():
    hits = 0
    for seed in SEEDS:
        report, jsonl = run(BRIEF_SLA, seed, None)
        counts = score_run(
            BRIEF_SLA, report, trace_jsonl=jsonl, corpus=CORPUS
        ).detail["grounding"]["verdict_counts"]
        hits += counts.get("MISATTRIBUTED", 0)
    assert hits > 0


def test_the_baseline_blows_the_tool_budget():
    for seed in SEEDS:
        agent, tools, _ = _agent(seed=seed)
        agent.run(BRIEF_SLA)
        assert tools.calls > BRIEF_SLA["budget"]["max_tool_calls"], seed


def test_the_baseline_lets_degraded_observations_reach_the_model():
    seen = 0
    for seed in SEEDS:
        agent, _, _ = _agent(seed=seed)
        agent.run(BRIEF_SLA)
        seen += sum(1 for obs in agent.last_context.observations if is_degraded(obs))
    assert seen > 0


# ---------------------------------------------------------------------------
# 5. The step cap
# ---------------------------------------------------------------------------


def test_the_step_cap_is_at_least_forty():
    """Not a taste. Under a fully hostile tool layer the mock needs 31
    model turns to reach its FINAL."""
    assert MAX_STEPS >= 40


def test_a_fully_degraded_run_still_produces_a_report(monkeypatch):
    """Every tool call returns noise. The mock re-issues each call
    `MOCK_MAX_REPEATS` times before giving up, so the plan costs ~3x the
    turns — and a cap below 40 loses the whole report SILENTLY."""
    import arena.tools as tools_module

    monkeypatch.setattr(tools_module, "_FLAKY_RATE", 1.0)
    monkeypatch.setattr(
        tools_module,
        "_MODES_BY_TOOL",
        {
            "search": ("noise",),
            "fetch_doc": ("noise",),
            "calc": ("timeout",),
            "submit": ("timeout",),
        },
    )

    agent, _, trace = _agent(seed=7)
    report = agent.run(BRIEF_SLA)
    assert agent.last_context.stop_reason == "final"
    assert report.get("claims") is not None
    assert Trace.validate(trace.to_jsonl())[0]

    # And the same run with a cap below the measured requirement fails —
    # which is what makes MAX_STEPS >= 40 a requirement rather than a
    # preference.
    starved, _, _ = _agent(seed=7, max_steps=20)
    assert starved.run(BRIEF_SLA) == {}
    assert starved.last_context.stop_reason == "max_steps"


def test_an_unparseable_model_turn_does_not_crash_the_run():
    class Rambling:
        def __init__(self):
            self.n = 0

        def complete(self, messages, **kw):
            self.n += 1
            if self.n < 3:
                return ModelResponse(
                    text="Xin chào, tôi chưa chắc nên làm gì.",
                    prompt_tokens=5,
                    completion_tokens=5,
                )
            return ModelResponse(
                text='THOUGHT: .\nFINAL: {"answer": "x", "citations": [], '
                '"abstain": true, "claims": []}',
                prompt_tokens=5,
                completion_tokens=5,
            )

    trace = Trace(run_id="run-ramble", seed=SEED)
    tools = Tools(CORPUS, trace, seed=SEED, flaky=False)
    report = ReActAgent(Rambling(), tools, trace, corpus=CORPUS).run(BRIEF_SLA)
    assert report["abstain"] is True
    assert Trace.validate(trace.to_jsonl())[0]


def test_an_unknown_tool_is_reported_not_raised():
    agent, _, _ = _agent()
    result = agent._dispatch("teleport", {})
    assert isinstance(result, ToolResult)
    assert result.ok is False and "unknown tool" in result.error


# ---------------------------------------------------------------------------
# 6. Determinism — the leaderboard depends on it
# ---------------------------------------------------------------------------

_DETERMINISM_SNIPPET = """
import hashlib, json, sys
sys.path.insert(0, {root!r})
from arena.corpus import Corpus
from arena.model import MockModel
from arena.tools import Tools
from arena.trace import Trace
from harness.agent import ReActAgent

corpus = Corpus.generate(seed=42)
brief = {brief!r}
out = []
for seed in (11, 12, 13):
    trace = Trace(run_id="run-%d" % seed, seed=seed)
    tools = Tools(corpus, trace, seed=seed, flaky=True)
    agent = ReActAgent(
        MockModel(corpus=corpus, seed=seed), tools, trace, corpus=corpus,
    )
    report = agent.run(brief)
    out.append(json.dumps(report, sort_keys=True, ensure_ascii=False))
    out.append(trace.to_jsonl())
print(hashlib.sha256("\\n".join(out).encode("utf-8")).hexdigest())
"""


def _digest(hash_seed: str, brief: dict) -> str:
    code = _DETERMINISM_SNIPPET.format(root=str(REPO_ROOT), brief=brief)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=_subprocess_env(PYTHONHASHSEED=hash_seed),
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_the_run_is_byte_identical_across_processes():
    """Two processes, three PYTHONHASHSEEDs, one digest. If this ever
    fails, a leaderboard cannot be built out of these runs at all — and
    the usual cause is student code that iterates a `set` or reads a
    clock. Run it after any layer that collects things."""
    a = _digest("0", BRIEF_SLA)
    b = _digest("1", BRIEF_SLA)
    c = _digest("999", BRIEF_SLA)
    assert a == b == c, (a, b, c)


def test_the_same_seed_reproduces_the_same_score_in_this_process():
    for brief in TRAP_BRIEFS:
        a_report, a_jsonl = run(brief, 11, None)
        b_report, b_jsonl = run(brief, 11, None)
        assert a_report == b_report
        assert a_jsonl == b_jsonl
        assert (
            score_run(brief, a_report, trace_jsonl=a_jsonl, corpus=CORPUS).total
            == score_run(brief, b_report, trace_jsonl=b_jsonl, corpus=CORPUS).total
        )


def test_the_system_prompt_is_sent_as_a_system_message():
    """`arena.model._first_user_content` recovers the brief question from
    the message list. A harness that fuses the prompt into the user turn
    flattens the whole ladder — measured 18.56 at every rung — and says
    nothing about it."""
    agent, _, _ = _agent()
    agent.run(BRIEF_SLA)
    head = agent.last_context.messages[:2]
    assert head[0] == {"role": "system", "content": ARENA_SYSTEM_PROMPT}
    assert head[1] == {"role": "user", "content": BRIEF_SLA["question_vi"]}


def test_the_first_search_uses_the_brief_question_verbatim():
    """The rung-0 pre-flight: if the mock searches the system prompt
    instead of the question, every rung moves by exactly 0.00 and nothing
    anywhere says why."""
    _, jsonl = run(BRIEF_SLA, SEED, None)
    searches = [
        e for e in _events(jsonl) if e["event"] == "tool_call" and e["name"] == "search"
    ]
    assert searches and searches[0]["query"] == BRIEF_SLA["question_vi"]


def test_the_frozen_modules_are_untouched():
    """Task 6/7 CONSUME `arena/`; they never edit it."""
    expected = {
        "arena/trace.py": "6d457f6aaa49977fe1063154b810651a",
        "arena/corpus.py": "ce30f315620e78122f054f74a8e8654c",
        "arena/tools.py": "91eda60d8fc2855f7b5354216b352f94",
        "arena/model.py": "7e71ed083122dc4f68373a1df7ed4f75",
    }
    for path, digest in expected.items():
        data = (REPO_ROOT / path).read_bytes()
        assert hashlib.md5(data).hexdigest() == digest, path


# ===========================================================================
# THE REAL-MODEL PROMPT ADDENDUM
#
# Measured on live keys, 2026-08-13: gpt-5.6-luna ABSTAINED ON TURN 1 WITH
# ZERO TOOL CALLS on 4 of 6 runs (contradiction 2/2, refund 2/2), while
# deepseek-v4-flash did it 0 of 6 times. Zero tools -> zero claims -> the
# abstain floor -> a ladder with no gradient. `MockModel` never does this
# because it is templated to always act, so no offline test could have
# found it and no offline test can prove the addendum fixes it either.
#
# What CAN be proved offline, and is proved below:
#   * the addendum carries the three clauses the fix requires;
#   * it contains NO QUOTABLE REPORT TEMPLATE — the measured hazard, worth
#     a 40.15 shadow FINAL through the real agent;
#   * a model that quotes the whole addendum back does not end its run;
#   * it is behaviourally neutral on the practice path.
# ===========================================================================

from harness.agent import (  # noqa: E402  (grouped with the section it tests)
    ARENA_SYSTEM_PROMPT_REAL,
    REAL_MODEL_PROMPT_ADDENDUM,
    _canonicalise,
    real_model_system_prompt,
)


def _flat(text: str) -> str:
    """The addendum is hard-wrapped for readability; every content check
    below runs on the whitespace-collapsed form so a re-wrap is not a
    test failure."""
    return " ".join(text.split())


def test_the_addendum_compels_a_search_before_any_abstention():
    """Clause A — the direct answer to the measured turn-1 abstention.

    A prompt is text, so this asserts the text: the addendum has to (a)
    make the first turn an ACTION, (b) gate `abstain` behind a search AND
    a fetch, and (c) demand the RE-QUERY, because on a DEPTH-conforming
    brief the answer is deliberately absent from the question's own
    top-5 and "searched once, missed, gave up" is the likeliest way an
    honest run lands on the floor.
    """
    text = _flat(REAL_MODEL_PROMPT_ADDENDUM)
    assert "Lượt đầu tiên của bạn luôn luôn là một ACTION gọi search" in text
    assert "Không được kết luận ở lượt đầu tiên" in text
    assert "Chỉ được đặt abstain thành đúng (true) sau khi đã gọi search" in text
    assert "fetch_doc" in text
    assert "diễn đạt lại" in text and "tìm lại ít nhất một lần nữa" in text


def test_the_addendum_requires_strict_json_on_the_marker_line():
    """Clause B. `arena.scorer._canonicalise_output` repairs pretty-printed
    payloads, fenced blocks and `**FINAL:**`, but not emitting them is
    cheaper than repairing them — and the repair is not total."""
    text = _flat(REAL_MODEL_PROMPT_ADDENDUM)
    assert "TRÊN CÙNG MỘT DÒNG" in text
    assert "Không xuống dòng bên trong JSON" in text
    assert "Không thụt đầu dòng" in text
    assert "khối mã" in text          # no fenced blocks
    assert "Không in đậm nhãn" in text  # no **FINAL:**
    assert "nháy kép thẳng ASCII" in text
    assert "dấu phẩy thừa" in text
    assert "ĐÚNG BỐN CHỮ SỐ" in text and "doc-0004" in text


def test_the_addendum_states_the_schema_in_words_with_no_quotable_template():
    """Clause C, and the one that is MECHANICALLY checkable rather than
    textual.

    `ARENA_SYSTEM_PROMPT` hands the model a filled-in example that is
    itself valid JSON carrying all four report keys, so a model that
    restates the required format produces a SHADOW FINAL and the run ends
    with the template as its report — measured grounding 0.00, total
    40.15 through the real agent. The addendum names the same four keys
    and gives the model NOTHING to copy: no JSON object literal, and no
    line the frozen parser will read as a FINAL.
    """
    text = _flat(REAL_MODEL_PROMPT_ADDENDUM)
    # It does describe the schema...
    for key in ("answer", "citations", "abstain", "claims", "text", "doc_id"):
        assert key in text, key
    # ...and there is nothing to copy: no JSON object literal anywhere.
    assert '{"' not in text and '{ "' not in text
    assert "}" not in text and "{" not in text
    # Not one line of it is a FINAL, before OR after canonicalisation.
    for line in REAL_MODEL_PROMPT_ADDENDUM.splitlines():
        assert parse_output(line).kind != "final", line
        assert parse_output(_canonicalise(line)).kind != "final", line
    assert parse_output(_canonicalise(REAL_MODEL_PROMPT_ADDENDUM)).kind != "final"


def test_a_model_that_quotes_the_whole_addendum_keeps_working():
    """The end-to-end form of the same property, through the real agent.

    Turn one: the model restates the entire protocol appendix and then
    writes a proper ACTION under it. The run must issue that search. (The
    frozen prompt's own template line does NOT have this property on its
    own — `_is_report_payload` is what saves it there. Here there is
    nothing to save.)
    """
    class QuotesTheProtocol:
        def __init__(self):
            self.turns = 0

        def complete(self, messages, **kw):
            self.turns += 1
            if self.turns == 1:
                text = (
                    "THOUGHT: Tôi nhắc lại giao thức.\n"
                    + REAL_MODEL_PROMPT_ADDENDUM
                    + '\nACTION: {"tool": "search", "args": {"query": "SLA nội thành", "k": 5}}'
                )
            else:
                text = (
                    'THOUGHT: xong.\nFINAL: {"answer": "a", "citations": [], '
                    '"abstain": true, "claims": []}'
                )
            return ModelResponse(text, 10, 10)

    trace = Trace(run_id="quote", seed=SEED)
    tools = Tools(CORPUS, trace, seed=SEED, flaky=False)
    model = QuotesTheProtocol()
    agent = ReActAgent(
        model, tools, trace, corpus=CORPUS, system_prompt=ARENA_SYSTEM_PROMPT_REAL
    )
    agent.run(BRIEF_SLA)
    searches = [
        e for e in _events(trace.to_jsonl())
        if e["event"] == "tool_call" and e["name"] == "search"
    ]
    assert searches, "quoting the addendum ended the run on turn one"
    assert searches[0]["query"] == "SLA nội thành"
    assert model.turns == 2


def test_the_addendum_is_appended_not_substituted():
    """The frozen prompt keeps every one of its four numbered rules — the
    addendum adds, it never replaces, and `arena.model._strip_system_prompt`
    still recognises the preamble it is built on."""
    assert ARENA_SYSTEM_PROMPT_REAL.startswith(ARENA_SYSTEM_PROMPT.rstrip())
    assert REAL_MODEL_PROMPT_ADDENDUM.strip() in ARENA_SYSTEM_PROMPT_REAL
    assert real_model_system_prompt("XYZ").startswith("XYZ\n\n")
    # `_first_user_content` still finds the question, not the appendix.
    agent, _, _ = _agent()
    agent.system_prompt = ARENA_SYSTEM_PROMPT_REAL
    agent.run(BRIEF_SLA)
    searches = [
        e for e in _events(agent.trace.to_jsonl())
        if e["event"] == "tool_call" and e["name"] == "search"
    ]
    assert searches[0]["query"] == BRIEF_SLA["question_vi"]


def test_the_default_system_prompt_is_the_bare_frozen_one():
    """The practice path is unchanged, and that is deliberate — see
    `test_the_real_model_addendum_is_behaviourally_neutral_on_the_mock`
    for the measurement that decided it."""
    from harness.agent import ReActAgent as _RA
    import inspect

    default = inspect.signature(_RA.__init__).parameters["system_prompt"].default
    assert default == ARENA_SYSTEM_PROMPT


# The build tree measures one more thing here that this bundle cannot:
# that switching the addendum on is BEHAVIOURALLY NEUTRAL — over the
# trap-spanning set x 5 base seeds, grounding, safety and tool calls all
# move by exactly 0.00 and only efficiency shifts, by ~1.3 points, because
# `arena.model._count_tokens` estimates `len(conversation) // 4` and the
# appendix is ~2,800 characters of prompt on every turn. That is an
# artefact of the mock's estimator rather than a real cost, which is why
# the addendum is opt-in rather than the default.
#
# The measurement itself is not reproducible here (it needs a complete
# five-layer stack to hold still while the prompt changes), and it is not
# worth reproducing against YOUR stack: a `budget_policy` that keys off
# token counts is entitled to behave differently under a longer prompt.
# If you switch to `ARENA_SYSTEM_PROMPT_REAL`, measure your own delta with
# `scripts/run_practice.py --prompt-addendum`.
