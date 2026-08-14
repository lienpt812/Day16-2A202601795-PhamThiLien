"""Acceptance tests for the FROZEN runner (`arena/runner.py`).

The runner exists for one reason: `arena/scorer.py` credits a claim only
if its text appears in the parsed FINAL payload of some `model_call`
event's `output_text`, so whoever controls `output_text` controls the
score. Everything below is organised around the three clauses that make
that field mean something, and each of the three is tested with a HOSTILE
middleware rather than with a friendly assertion — a passive check that
the field "looks right" would pass against a harness that rewrites it.

    1. `output_text` is the RAW client response, captured inside the
       frozen wrapper, before any student hook runs.
    2. Once a `model_call` exists, `output_text` is present and non-empty.
    3. The runner records what it ACTUALLY sent (`prompt_sha256`).

Plus the measured requirements around them: normalisation happens BEFORE
the stamp (not merely inside the runner's parser), the report is read off
the `submit` event, `max_tokens` is 3000 with automatic parameter
fallback, a preflight fails loudly, a run with no FINAL becomes an
explicit abstention, and the caps cut a hung harness without breaking the
gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import pytest

from arena.briefs import load_public_briefs
from arena.corpus import Corpus
from arena.model import (
    ARENA_SYSTEM_PROMPT,
    BaseModel,
    MockModel,
    ModelResponse,
    RealModel,
    RealModelError,
    parse_output,
)
from arena.runner import (
    DEFAULT_MAX_TOKENS,
    EMPTY_OUTPUT_SENTINEL,
    RUNNER_VERSION,
    PreflightError,
    ProvenanceModel,
    RunAborted,
    RunnerConfig,
    _Guard,
    assert_provenance,
    clamp_output_text,
    derive_seed,
    normalise_output,
    preflight,
    run_brief,
    run_session,
    score_result,
    strip_timing,
    synthetic_abstain_report,
)


def _subprocess_env(**extra):
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.update(extra)
    return env
from arena.scorer import MODEL_OUTPUT_FIELD, score_run
from arena.tools import Tools
from arena.trace import Trace

from harness.agent import ReActAgent
from harness.middleware import Middleware


def student_stack():
    """The five layers from `harness/layers/`, in the documented order.

    On a fresh checkout every one of them is an unedited stub, so this is
    the BASELINE with five no-ops installed — and that is exactly the
    property several tests below want: the runner must cost your stack
    nothing, whether or not you have written any of it yet.
    """
    from harness.layers.budget_policy import BudgetPolicy
    from harness.layers.citation_checker import CitationChecker
    from harness.layers.critic import Critic
    from harness.layers.injection_guard import InjectionGuard
    from harness.layers.retry import Retry

    return [InjectionGuard(), Critic(), CitationChecker(), BudgetPolicy(), Retry()]

LAB_ROOT = Path(__file__).resolve().parent.parent
SEED = 42
CORPUS = Corpus.generate(seed=SEED)
BRIEFS = load_public_briefs()
BRIEF = BRIEFS[0]
QUIET = RunnerConfig(warn_on_missing_final=False)

#: A verbatim line of doc-0004 — the SLA document the first public brief
#: is about. Read out of the corpus, never retyped, so a corpus-seed
#: change fails loudly instead of silently.
SUPPORTED_LINE = next(
    line.strip()
    for line in CORPUS.get("doc-0004").body.splitlines()
    if "nội thành 2 ngày" in line
)

REPORT_PAYLOAD = {
    "answer": "Theo tài liệu: " + SUPPORTED_LINE,
    "citations": ["doc-0004"],
    "abstain": False,
    "claims": [{"text": SUPPORTED_LINE, "doc_id": "doc-0004"}],
}


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def mock_run(middleware=None, seed=11, brief=BRIEF, config=QUIET, model=None):
    return run_brief(
        brief,
        model=model if model is not None else MockModel(corpus=CORPUS, seed=seed),
        corpus=CORPUS,
        middleware=middleware,
        seed=seed,
        config=config,
    )


def records(jsonl: str, event: str | None = None) -> list:
    out = []
    for line in jsonl.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if event is None or record.get("event") == event:
            out.append(record)
    return out


def outputs(result) -> list:
    return [call[MODEL_OUTPUT_FIELD] for call in records(result.trace_jsonl, "model_call")]


class ScriptedModel(BaseModel):
    """Returns fixed text. The way to test one output shape end to end."""

    def __init__(self, text, prompt_tokens=20, completion_tokens=20):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.calls: list = []

    def complete(self, messages, **kw):
        self.calls.append(([dict(m) for m in messages], dict(kw)))
        text = self.text(len(self.calls)) if callable(self.text) else self.text
        return ModelResponse(
            text=text,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )


def _search_then(text: str):
    """A model that RETRIEVES first and then writes `text`.

    The scorer will not credit a citation to a document the run never
    looked at (`UNRETRIEVED`), so a shape test that answered on turn one
    would measure retrieval, not shape handling.
    """
    search = (
        'THOUGHT: tìm\nACTION: {"tool": "search", "args": {"query": '
        + json.dumps(BRIEF["question_vi"], ensure_ascii=False)
        + ', "k": 5}}'
    )
    return ScriptedModel(lambda turn: search if turn == 1 else text)


@lru_cache(maxsize=None)
def _ladder(kind: str) -> tuple:
    """Mean total over the public set, through the runner.

    `kind` is "baseline" (no middleware) or "student" (your five layers).
    """
    totals = []
    for index, brief in enumerate(BRIEFS):
        seed = derive_seed(11, index)
        layers = student_stack() if kind == "student" else None
        result = run_brief(
            brief,
            model=MockModel(corpus=CORPUS, seed=seed),
            corpus=CORPUS,
            middleware=layers,
            seed=seed,
            config=QUIET,
        )
        totals.append(score_result(result, brief, CORPUS).total)
    return tuple(totals)


# ---------------------------------------------------------------------------
# 1. The contract: what the runner writes
# ---------------------------------------------------------------------------


def test_a_baseline_run_produces_a_conforming_trace():
    result = mock_run()
    ok, reason = result.gate()
    assert ok, reason
    assert score_result(result, BRIEF, CORPUS).gate_passed


def test_the_trace_starts_with_agent_start_and_ends_with_agent_end():
    events = [record["event"] for record in records(mock_run().trace_jsonl)]
    assert events[0] == "agent_start"
    assert events[-1] == "agent_end"


def test_agent_end_carries_elapsed_seconds():
    """Efficiency's wall-clock component reads this, and the harness must
    not be the one to write it — a clock inside student code is both
    non-deterministic and forgeable."""
    result = mock_run()
    ends = records(result.trace_jsonl, "agent_end")
    assert any("elapsed_seconds" in end for end in ends)
    last = ends[-1]
    assert last["elapsed_seconds"] >= 0.0
    assert last["runner"] == RUNNER_VERSION
    assert score_run(BRIEF, result.report, result.trace_jsonl, CORPUS).gate_passed


def test_the_runner_scores_the_report_recorded_on_the_submit_event():
    """The object `run()` returns and the object it SUBMITTED can differ.

    The scorer refuses any claim absent from the submitted report
    (`NOT_SUBMITTED`), so the submitted one is the one to score.
    """

    class SubmitsSomethingElse(Middleware):
        name = "sneaky"

        def after_agent(self, ctx, report):
            ctx.tools.submit({"answer": "ĐÃ NỘP", "claims": [], "citations": [],
                              "abstain": True})
            return dict(report, answer="ĐÃ TRẢ VỀ")

    result = mock_run(middleware=[SubmitsSomethingElse()])
    assert result.report_source == "submit_event"
    # The agent's own submit runs after the layer's, so the LAST submit
    # wins — and that is the one the scorer cross-checks against.
    submits = records(result.trace_jsonl, "tool_call")
    submitted = [json.loads(r["report_json"]) for r in submits if r.get("name") == "submit"]
    assert result.report == submitted[-1]


def test_the_runner_path_scores_identically_to_a_plain_run():
    """THE false-positive check: the runner must cost an honest run ZERO.

    Same brief, same seed, same layers — once through a plain
    `Trace`/`Tools`/agent and once through the frozen runner. Any
    difference means the runner is charging honest students for
    something.
    """
    for make_layers in (lambda: None, student_stack):
        plain_totals, runner_totals = [], []
        for index, brief in enumerate(BRIEFS):
            seed = derive_seed(11, index)
            trace = Trace(run_id=f"{brief['brief_id']}-{seed}", seed=seed)
            tools = Tools(CORPUS, trace, seed=seed, flaky=True)
            agent = ReActAgent(
                MockModel(corpus=CORPUS, seed=seed), tools, trace,
                middleware=make_layers(), corpus=CORPUS,
            )
            report = agent.run(brief)
            plain_totals.append(
                score_run(brief, report, trace.to_jsonl(), CORPUS).total
            )
            result = run_brief(
                brief, model=MockModel(corpus=CORPUS, seed=seed), corpus=CORPUS,
                middleware=make_layers(), seed=seed, config=QUIET,
            )
            runner_totals.append(score_result(result, brief, CORPUS).total)
        assert plain_totals == runner_totals


def test_per_brief_seeds_are_derived():
    """One seed for a whole set replays the IDENTICAL failure pattern on
    every brief — `arena/tools.py` keys flakiness on (seed, call index).
    """
    assert derive_seed(11, 0) == 11
    assert derive_seed(11, 3) == 14
    results = run_session(
        BRIEFS[:3],
        model_factory=lambda seed: MockModel(corpus=CORPUS, seed=seed),
        corpus=CORPUS,
        base_seed=11,
        config=QUIET,
    )
    assert [r.seed for r in results] == [11, 12, 13]


# ---------------------------------------------------------------------------
# 2. CLAUSE 1 — output_text is the RAW client response
# ---------------------------------------------------------------------------

POISON = "ĐÂY-LÀ-CHỮ-DO-HARNESS-VIẾT"


def test_a_hostile_after_model_cannot_rewrite_output_text():
    class Hostile(Middleware):
        name = "hostile-after"

        def after_model(self, ctx, response):
            return ModelResponse(text=POISON, prompt_tokens=0, completion_tokens=0)

    result = mock_run(
        middleware=[Hostile()],
        config=RunnerConfig(max_model_calls=3, warn_on_missing_final=False),
    )
    assert outputs(result), "no model_call was recorded at all"
    assert not any(POISON in text for text in outputs(result))
    assert all(text.startswith("THOUGHT:") for text in outputs(result))


def test_a_hostile_wrap_model_call_cannot_rewrite_output_text():
    class Hostile(Middleware):
        name = "hostile-wrap"

        def wrap_model_call(self, ctx, call, messages):
            call(messages)  # the real call still happens and is recorded
            return ModelResponse(text=POISON, prompt_tokens=999, completion_tokens=999)

    result = mock_run(
        middleware=[Hostile()],
        config=RunnerConfig(max_model_calls=3, warn_on_missing_final=False),
    )
    assert not any(POISON in text for text in outputs(result))


def test_emptying_the_response_cannot_mute_the_provenance_rule():
    """The measured MUTE bypass: a hook sets `response.text = ""` so the
    scorer sees no recorded model output and disables the rule. Stamping
    inside the client wrapper makes it inert."""

    class Mute(Middleware):
        name = "mute"

        def after_model(self, ctx, response):
            return ModelResponse(
                text=response.text, prompt_tokens=0, completion_tokens=0
            )

    class HardMute(Middleware):
        name = "hard-mute"

        def after_model(self, ctx, response):
            return ModelResponse(text="", prompt_tokens=0, completion_tokens=0)

    quiet = mock_run(middleware=[Mute()])
    muted = mock_run(
        middleware=[HardMute()],
        config=RunnerConfig(max_model_calls=6, warn_on_missing_final=False),
    )
    assert all(outputs(muted)), "an emptied response produced an empty output_text"
    detail = score_result(quiet, BRIEF, CORPUS).detail
    assert detail["provenance"]["model_text_recorded"] is True


def test_a_layer_mutating_the_response_object_cannot_move_the_record():
    """`ModelResponse` is a frozen dataclass, but a determined layer can
    still `object.__setattr__` it. The trace holds an immutable `str`
    captured before any of that."""

    class Mutator(Middleware):
        name = "mutator"

        def after_model(self, ctx, response):
            try:
                object.__setattr__(response, "text", POISON)
            except Exception:  # pragma: no cover - defensive
                pass
            return response

    result = mock_run(
        middleware=[Mutator()],
        config=RunnerConfig(max_model_calls=3, warn_on_missing_final=False),
    )
    assert not any(POISON in text for text in outputs(result))


def test_the_runner_not_the_agent_emits_model_call():
    """`harness/agent.py` stands down when the model announces
    `emits_model_call`; the guarded trace enforces it rather than
    trusting it."""
    result = mock_run()
    calls = records(result.trace_jsonl, "model_call")
    assert len(calls) == result.model_calls
    assert all("prompt_sha256" in call for call in calls), (
        "a model_call without the runner's own fields means student code wrote it"
    )


# ---------------------------------------------------------------------------
# 3. CLAUSE 2 — output_text is never empty once a model_call exists
# ---------------------------------------------------------------------------


def test_output_text_is_never_emitted_empty():
    class Silent(BaseModel):
        def complete(self, messages, **kw):
            return ModelResponse(text="", prompt_tokens=5, completion_tokens=0)

    result = run_brief(
        BRIEF, model=Silent(), corpus=CORPUS, seed=11,
        config=RunnerConfig(max_model_calls=3, warn_on_missing_final=False),
    )
    assert outputs(result)
    assert all(text == EMPTY_OUTPUT_SENTINEL for text in outputs(result))
    assert all(text for text in outputs(result))


def test_a_non_string_response_still_produces_a_non_empty_output_text():
    class Weird(BaseModel):
        def complete(self, messages, **kw):
            return ModelResponse(text=None, prompt_tokens=1, completion_tokens=1)

    result = run_brief(
        BRIEF, model=Weird(), corpus=CORPUS, seed=11,
        config=RunnerConfig(max_model_calls=2, warn_on_missing_final=False),
    )
    assert all(text for text in outputs(result))


@pytest.mark.parametrize("index", range(len(BRIEFS)))
def test_every_honest_run_records_non_empty_output_on_every_model_call(index):
    seed = derive_seed(11, index)
    result = run_brief(
        BRIEFS[index], model=MockModel(corpus=CORPUS, seed=seed), corpus=CORPUS,
        middleware=student_stack(), seed=seed, config=QUIET,
    )
    assert result.model_calls > 0
    assert all(text for text in outputs(result))
    assert result.provenance_ok


# ---------------------------------------------------------------------------
# 4. CLAUSE 3 — the runner records what it actually sent
# ---------------------------------------------------------------------------


def test_prompt_sha256_records_what_was_actually_sent():
    model = ScriptedModel('THOUGHT: x\nFINAL: {"abstain": true, "claims": []}')
    result = run_brief(BRIEF, model=model, corpus=CORPUS, seed=11, config=QUIET)
    calls = records(result.trace_jsonl, "model_call")
    assert len(calls) == len(model.calls)
    for call, (messages, _kw) in zip(calls, model.calls):
        snapshot = [
            {"role": m.get("role", ""), "content": m.get("content", "")}
            for m in messages
        ]
        blob = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        assert call["prompt_sha256"] == hashlib.sha256(blob.encode("utf-8")).hexdigest()
        assert call["prompt_chars"] == len(blob)
        assert call["n_messages"] == len(snapshot)


def test_a_layer_that_changes_the_prompt_changes_the_recorded_hash():
    class Nudger(Middleware):
        name = "nudger"

        def before_model(self, ctx, messages):
            return messages + [{"role": "user", "content": "Hãy trả lời ngay."}]

    plain = mock_run(config=RunnerConfig(max_model_calls=1, warn_on_missing_final=False))
    nudged = mock_run(
        middleware=[Nudger()],
        config=RunnerConfig(max_model_calls=1, warn_on_missing_final=False),
    )
    a = records(plain.trace_jsonl, "model_call")[0]["prompt_sha256"]
    b = records(nudged.trace_jsonl, "model_call")[0]["prompt_sha256"]
    assert a != b


def test_unaccounted_chars_separates_retrieved_evidence_from_pasted_evidence():
    """The number an instructor reads to answer "was this retrieved, or
    typed in?". Informational, never scored."""

    class Paster(Middleware):
        name = "paster"

        def before_model(self, ctx, messages):
            body = ctx.corpus.get("doc-0004").body
            return messages + [{"role": "user", "content": "Bằng chứng:\n" + body}]

    config = RunnerConfig(max_model_calls=1, warn_on_missing_final=False)
    plain = records(mock_run(config=config).trace_jsonl, "model_call")[0]
    pasted = records(
        mock_run(middleware=[Paster()], config=config).trace_jsonl, "model_call"
    )[0]
    assert plain["unaccounted_chars"] == 0
    assert pasted["unaccounted_chars"] > 200


def test_a_tool_observation_is_accounted_for_and_not_flagged_as_pasted():
    """The stock harness feeds observations back into the prompt. Those
    are RETRIEVED, so they must not read as harness-authored text."""
    result = mock_run(config=RunnerConfig(max_model_calls=4, warn_on_missing_final=False))
    calls = records(result.trace_jsonl, "model_call")
    assert len(calls) >= 3
    assert all(call["unaccounted_chars"] == 0 for call in calls)


# ---------------------------------------------------------------------------
# 5. Normalisation happens BEFORE the stamp. Measured placement.
# ---------------------------------------------------------------------------

_PAYLOAD_JSON = json.dumps(REPORT_PAYLOAD, ensure_ascii=False)
_PRETTY = json.dumps(REPORT_PAYLOAD, ensure_ascii=False, indent=2)

FINAL_SHAPES = {
    "canonical": "THOUGHT: đủ rồi\nFINAL: " + _PAYLOAD_JSON,
    "leading-spaces": "  THOUGHT: ok\n    FINAL: " + _PAYLOAD_JSON,
    "leading-tab": "\tFINAL: " + _PAYLOAD_JSON,
    "bold-marker": "**FINAL:** " + _PAYLOAD_JSON,
    "lowercase": "final: " + _PAYLOAD_JSON,
    "title-case": "Final: " + _PAYLOAD_JSON,
    "space-before-colon": "FINAL : " + _PAYLOAD_JSON,
    "heading": "### FINAL\n" + _PAYLOAD_JSON,
    "bullet": "- FINAL: " + _PAYLOAD_JSON,
    "blockquote": "> FINAL: " + _PAYLOAD_JSON,
    "fenced": "Kết quả:\n```json\nFINAL: " + _PAYLOAD_JSON + "\n```",
    "fenced-no-marker": "Kết quả:\n```json\n" + _PAYLOAD_JSON + "\n```",
    "bom": "﻿FINAL: " + _PAYLOAD_JSON,
    "bom-crlf": "﻿THOUGHT: ok\r\nFINAL: " + _PAYLOAD_JSON + "\r\n",
    "pretty-next-line": "FINAL:\n" + _PRETTY,
    "pretty-same-line": "FINAL: " + _PRETTY,
    "payload-next-line": "FINAL:\n" + _PAYLOAD_JSON,
    "bare-json": _PAYLOAD_JSON,
    "prose-then-json": "Tôi đã tìm thấy câu trả lời.\n" + _PAYLOAD_JSON,
    "smart-quotes": "FINAL: " + _PAYLOAD_JSON.replace('"', "”"),
    "trailing-comma": "FINAL: " + _PAYLOAD_JSON[:-1] + ",}",
    "list-wrapped": "FINAL: [" + _PAYLOAD_JSON + "]",
    "trailing-prose": "FINAL: " + _PAYLOAD_JSON + "\nHy vọng hữu ích.",
    "thought-final-prose": (
        "THOUGHT: xong\nFINAL: " + _PAYLOAD_JSON + "\n\nNếu cần thêm, hãy hỏi."
    ),
    "indented-fence": "```\n   final : " + _PAYLOAD_JSON + "\n```",
}

ACTION_SHAPES = {
    "canonical": 'THOUGHT: tìm\nACTION: {"tool": "search", "args": {"query": "sla", "k": 5}}',
    "pretty": 'THOUGHT: tìm\nACTION:\n{\n  "tool": "search",\n  "args": {"query": "sla", "k": 5}\n}',
    "fenced": 'THOUGHT: tìm\n```json\nACTION: {"tool": "search", "args": {"query": "sla"}}\n```',
    "lowercase": 'action: {"tool": "search", "args": {"query": "sla"}}',
    "indented": '   ACTION: {"tool": "search", "args": {"query": "sla"}}',
}


@pytest.mark.parametrize("name", sorted(FINAL_SHAPES))
def test_the_stamped_output_text_parses_as_a_final_for_every_real_shape(name):
    """The measured requirement, checked where it matters: on the field
    the SCORER reads, not on the runner's own parse. Repairing only the
    parser moved shape coverage 56.8% -> 60.0%; repairing before the
    stamp moved it to 76.8%."""
    model = ScriptedModel(FINAL_SHAPES[name])
    result = run_brief(BRIEF, model=model, corpus=CORPUS, seed=11, config=QUIET)
    stamped = outputs(result)
    assert stamped, name
    assert parse_output(stamped[0]).kind == "final", (name, stamped[0][:200])
    assert result.final_outputs >= 1


@pytest.mark.parametrize("name", sorted(FINAL_SHAPES))
def test_a_claim_is_credited_end_to_end_through_every_real_shape(name):
    """Not just "the shape parses" — the claim must survive all the way
    to a `SUPPORTED` verdict, which is the thing worth 55 points."""
    result = run_brief(
        BRIEF, model=_search_then(FINAL_SHAPES[name]), corpus=CORPUS, seed=11,
        config=RunnerConfig(flaky=False, warn_on_missing_final=False),
    )
    score = score_result(result, BRIEF, CORPUS)
    counts = score.detail.get("grounding", {}).get("verdict_counts", {})
    assert counts.get("SUPPORTED"), (name, counts)
    assert not counts.get("NOT_FROM_MODEL"), (name, counts)


@pytest.mark.parametrize("name", sorted(ACTION_SHAPES))
def test_an_action_shape_is_never_turned_into_a_final(name):
    """Normalisation must never end a run that was still working."""
    normalised = normalise_output(ACTION_SHAPES[name])
    parsed = parse_output(normalised)
    assert parsed.kind == "action", (name, normalised)
    assert parsed.tool == "search"


def test_a_quoted_protocol_template_does_not_shadow_the_action_underneath():
    """A model quoting `ARENA_SYSTEM_PROMPT`'s own template writes a
    payload with all four report keys and `"..."` in every slot. The
    normaliser must keep the ACTION line so the agent's guard can find
    it."""
    text = (
        'FINAL: {"answer": "...", "citations": ["doc-0001"], "abstain": false, '
        '"claims": [{"text": "...", "doc_id": "doc-0001"}]}\n'
        'ACTION: {"tool": "search", "args": {"query": "sla", "k": 5}}'
    )
    normalised = normalise_output(text)
    assert "ACTION:" in normalised
    lines = normalised.splitlines()
    assert lines[0].startswith("FINAL:") and lines[1].startswith("ACTION:")


def test_normalisation_is_a_no_op_on_canonical_mock_output():
    """The mock already writes the canonical protocol. If normalisation
    moved a single byte of it, every ladder number measured against the
    plain path would stop reproducing."""
    trace = Trace("t", 11)
    tools = Tools(CORPUS, trace, seed=11, flaky=False)
    agent = ReActAgent(MockModel(corpus=CORPUS, seed=11), tools, trace, corpus=CORPUS)
    agent.run(BRIEF)
    messages = agent.last_context.messages
    turns = [m["content"] for m in messages if m.get("role") == "assistant"]
    assert turns
    for text in turns:
        assert normalise_output(text) == text


def test_normalisation_never_invents_a_payload_the_scorer_would_not_have_found():
    """The append clause may only re-serialise payloads the scorer's own
    recogniser already accepts. Plain prose gets nothing."""
    for text in (
        "Tôi không tìm thấy tài liệu nào.",
        "THOUGHT: tôi còn phân vân.",
        "Câu trả lời là 2 ngày làm việc.",
    ):
        assert normalise_output(text) == text
        assert parse_output(normalise_output(text)).kind == "unparseable"


def test_clamping_keeps_the_final_payload():
    """A naive head-clamp cuts the FINAL in half, which stops it being
    JSON — 55 points, silently. Measured cliff: 88,422 chars fine, 90,422
    dead."""
    padded = "x" * 80_000 + "\nFINAL: " + _PAYLOAD_JSON
    clamped = clamp_output_text(padded, limit=5_000)
    assert len(clamped) <= 5_000
    assert parse_output(clamped).kind == "final"
    assert parse_output(clamped).final["claims"][0]["text"] == SUPPORTED_LINE


def test_a_gigantic_model_output_still_yields_a_scoreable_run():
    model = ScriptedModel("y" * 200_000 + "\nFINAL: " + _PAYLOAD_JSON)
    result = run_brief(BRIEF, model=model, corpus=CORPUS, seed=11, config=QUIET)
    ok, reason = result.gate()
    assert ok, reason
    assert result.final_outputs >= 1
    detail = score_result(result, BRIEF, CORPUS).detail
    assert detail["provenance"]["truncated_model_calls"] == 0


@pytest.mark.parametrize(
    "name,text",
    [
        ("pseudo-marker prose", "\n".join(["Finally, câu trả lời là 2 ngày."] * 20_000)),
        ("pseudo-marker braces", "\n".join(["Finally, {gần} đúng 2 ngày."] * 20_000)),
        ("megabyte of junk", "z" * 1_000_000),
        ("many real finals", "\n".join(['FINAL: {"answer": "a", "claims": []}'] * 5_000)),
        ("deep brackets", "FINAL: " + "[" * 2_000 + "]" * 2_000),
    ],
    ids=["prose", "braces", "junk", "finals", "brackets"],
)
def test_normalisation_is_bounded_on_pathological_output(name, text):
    """A per-turn cost, so it must stay milliseconds even on hostile
    input. The scorer's marker pattern makes the colon optional, so every
    line beginning "Finally," is a probe candidate: unbounded, that
    measured 855 ms on a one-megabyte output."""
    started = time.perf_counter()
    normalised = normalise_output(text)
    clamp_output_text(normalised)
    assert time.perf_counter() - started < 2.0, name


def test_the_report_is_extracted_with_the_frozen_parser_only():
    """A lenient in-house parser builds a plausible report out of text the
    scorer will not recognise, and then EVERY claim reads
    `NOT_FROM_MODEL`. Whatever the runner acts on must be what
    `arena.model.parse_output` recovers from the stamped text."""
    for name, text in FINAL_SHAPES.items():
        model = ScriptedModel(text)
        result = run_brief(BRIEF, model=model, corpus=CORPUS, seed=11, config=QUIET)
        stamped = outputs(result)[0]
        recovered = parse_output(stamped)
        assert recovered.kind == "final", name
        assert result.report.get("claims") == recovered.final.get("claims"), name


# ---------------------------------------------------------------------------
# 6. The output budget
# ---------------------------------------------------------------------------


def test_the_runner_asks_for_three_thousand_output_tokens():
    model = ScriptedModel('THOUGHT: x\nFINAL: {"abstain": true, "claims": []}')
    run_brief(BRIEF, model=model, corpus=CORPUS, seed=11, config=QUIET)
    assert model.calls
    for _messages, kwargs in model.calls:
        assert kwargs.get("max_tokens") == DEFAULT_MAX_TOKENS == 3000


def test_the_budget_is_configurable():
    model = ScriptedModel('THOUGHT: x\nFINAL: {"abstain": true, "claims": []}')
    run_brief(
        BRIEF, model=model, corpus=CORPUS, seed=11,
        config=RunnerConfig(max_tokens=1234, warn_on_missing_final=False),
    )
    assert model.calls[0][1]["max_tokens"] == 1234


class _FakeEndpoint(RealModel):
    """A `RealModel` whose transport is replaced. NO NETWORK: `_post` is
    overridden, so nothing is ever opened."""

    def __init__(self, reject_max_tokens: bool):
        super().__init__(base_url="https://example.invalid/v1", api_key="k", model="m")
        self.reject_max_tokens = reject_max_tokens
        self.posted: list = []

    def _post(self, payload):
        self.posted.append(dict(payload))
        if self.reject_max_tokens and "max_tokens" in payload:
            raise RealModelError(
                "Gọi endpoint thất bại: HTTP Error 400: Unsupported parameter: "
                "'max_tokens' is not supported with this model. Use "
                "'max_completion_tokens' instead."
            )
        return {
            "choices": [{"message": {"content": 'THOUGHT: x\nFINAL: {"abstain": true}'}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }


def _client(model, **kw):
    guard = _Guard(deadline=None, max_model_calls=None, max_tool_calls=None,
                   clock=time.perf_counter)
    return ProvenanceModel(model, Trace("t", 1), guard=guard, **kw)


def test_the_ordinary_path_sends_max_tokens_through_the_frozen_client():
    endpoint = _FakeEndpoint(reject_max_tokens=False)
    client = _client(endpoint)
    client.complete([{"role": "user", "content": "hi"}])
    assert endpoint.posted[0]["max_tokens"] == DEFAULT_MAX_TOKENS
    assert "max_completion_tokens" not in endpoint.posted[0]
    assert client.param_used == "max_tokens"


def test_an_endpoint_that_rejects_max_tokens_falls_back_automatically():
    endpoint = _FakeEndpoint(reject_max_tokens=True)
    client = _client(endpoint)
    response = client.complete([{"role": "user", "content": "hi"}])
    assert response.text.startswith("THOUGHT:")
    assert client.param_used == "max_completion_tokens"
    assert endpoint.posted[-1]["max_completion_tokens"] == DEFAULT_MAX_TOKENS


def test_the_working_parameter_is_remembered_for_the_rest_of_the_session():
    endpoint = _FakeEndpoint(reject_max_tokens=True)
    client = _client(endpoint)
    for _ in range(3):
        client.complete([{"role": "user", "content": "hi"}])
    # One wasted probe, then three good calls — not one wasted per turn.
    rejected = [p for p in endpoint.posted if "max_tokens" in p]
    assert len(rejected) == 1
    assert len(endpoint.posted) == 4


def test_the_preferred_parameter_can_be_pinned():
    endpoint = _FakeEndpoint(reject_max_tokens=True)
    client = _client(endpoint, max_tokens_param="max_completion_tokens")
    client.complete([{"role": "user", "content": "hi"}])
    assert [set(p) & {"max_tokens", "max_completion_tokens"} for p in endpoint.posted] == [
        {"max_completion_tokens"}
    ]


def test_an_unrelated_endpoint_failure_is_not_mistaken_for_a_parameter_problem():
    class Broken(_FakeEndpoint):
        def _post(self, payload):
            raise RealModelError("Gọi endpoint thất bại: HTTP Error 401: Unauthorized")

    with pytest.raises(RealModelError, match="401"):
        _client(Broken(reject_max_tokens=False)).complete([{"role": "user", "content": "x"}])


def test_a_client_whose_complete_takes_no_keywords_still_works():
    class Minimal:
        def complete(self, messages):
            return ModelResponse(
                text='THOUGHT: x\nFINAL: {"abstain": true}', prompt_tokens=1,
                completion_tokens=1,
            )

    response = _client(Minimal()).complete([{"role": "user", "content": "x"}])
    assert response.text.startswith("THOUGHT:")


# ---------------------------------------------------------------------------
# 7. The preflight
# ---------------------------------------------------------------------------


class NeverFinal(BaseModel):
    def complete(self, messages, **kw):
        return ModelResponse(text="Tôi chưa chắc.", prompt_tokens=4, completion_tokens=4)


def test_preflight_fails_loudly_when_no_model_call_carries_a_final():
    with pytest.raises(PreflightError) as excinfo:
        preflight(
            BRIEF, model=NeverFinal(), corpus=CORPUS, seed=11,
            config=RunnerConfig(max_model_calls=2, warn_on_missing_final=False),
        )
    message = str(excinfo.value)
    assert "NOT_FROM_MODEL" in message
    assert "final_outputs=0" in message


def test_preflight_passes_on_an_honest_run():
    result = preflight(
        BRIEF, model=MockModel(corpus=CORPUS, seed=11), corpus=CORPUS, seed=11,
        config=QUIET,
    )
    assert result.provenance_ok
    assert result.gate()[0]


def test_assert_provenance_passes_when_at_least_one_run_recorded_a_final():
    good = mock_run()
    bad = run_brief(
        BRIEF, model=NeverFinal(), corpus=CORPUS, seed=11,
        config=RunnerConfig(max_model_calls=2, warn_on_missing_final=False),
    )
    assert_provenance([good, bad])
    with pytest.raises(PreflightError):
        assert_provenance([bad])
    with pytest.raises(PreflightError):
        assert_provenance([])


def test_strict_provenance_stops_a_session_on_the_first_bad_run():
    """"After each run, assert" — the SCORED setting. A batch that has
    started producing NOT_FROM_MODEL runs must stop, not finish."""
    seen = []
    with pytest.raises(PreflightError):
        run_session(
            BRIEFS[:3],
            model_factory=lambda seed: NeverFinal(),
            corpus=CORPUS,
            base_seed=11,
            config=RunnerConfig(max_model_calls=2, warn_on_missing_final=False),
            on_result=seen.append,
            strict_provenance=True,
        )
    assert len(seen) == 1, "the batch kept going after the first silent zero"

    seen.clear()
    run_session(
        BRIEFS[:3],
        model_factory=lambda seed: MockModel(corpus=CORPUS, seed=seed),
        corpus=CORPUS,
        base_seed=11,
        config=QUIET,
        on_result=seen.append,
        strict_provenance=True,
    )
    assert len(seen) == 3


def test_a_run_with_no_decodable_final_is_flagged():
    result = run_brief(
        BRIEF, model=NeverFinal(), corpus=CORPUS, seed=11,
        config=RunnerConfig(max_model_calls=2, warn_on_missing_final=False),
    )
    assert "no_final_output" in result.flags
    assert not result.provenance_ok


# ---------------------------------------------------------------------------
# 8. The synthesised abstention
# ---------------------------------------------------------------------------


def test_a_run_with_no_final_submits_an_explicit_abstention():
    result = run_brief(
        BRIEF, model=NeverFinal(), corpus=CORPUS, seed=11,
        config=RunnerConfig(max_model_calls=3, warn_on_missing_final=False),
    )
    assert result.report_source == "synthesised"
    assert result.report["abstain"] is True
    assert "synthesised_abstain" in result.flags
    submits = [
        json.loads(r["report_json"])
        for r in records(result.trace_jsonl, "tool_call")
        if r.get("name") == "submit" and r.get("report_json")
    ]
    assert submits[-1] == result.report, "the abstention was not recorded on the trace"


def test_the_synthesised_abstention_beats_silence():
    """An honest run that hit the step cap must not land below a
    one-character troll."""
    config = RunnerConfig(max_model_calls=3, warn_on_missing_final=False)
    silent = RunnerConfig(max_model_calls=3, synthesise_abstain=False,
                          warn_on_missing_final=False)
    with_synth = score_result(
        run_brief(BRIEF, model=NeverFinal(), corpus=CORPUS, seed=11, config=config),
        BRIEF, CORPUS,
    ).total
    without = score_result(
        run_brief(BRIEF, model=NeverFinal(), corpus=CORPUS, seed=11, config=silent),
        BRIEF, CORPUS,
    ).total
    assert without == 0.0
    assert with_synth > without


def test_the_synthesised_abstention_never_overwrites_a_real_report():
    result = mock_run()
    assert result.report_source == "submit_event"
    assert "synthesised_abstain" not in result.flags
    assert result.report.get("claims")


def test_a_layer_that_raises_is_not_rescued_by_the_synthesised_abstention():
    """"Your layer raises and you score zero" is the documented rule. The
    runner records the crash and keeps the batch alive; it does not hand
    the entry an abstention it never made."""

    class Exploder(Middleware):
        name = "exploder"

        def after_agent(self, ctx, report):
            raise ValueError("boom")

    result = mock_run(middleware=[Exploder()])
    assert "ValueError" in result.error
    assert result.report_source != "synthesised"
    assert score_result(result, BRIEF, CORPUS).total == 0.0
    assert result.gate()[0], "a crashed run must still leave a readable trace"


def test_the_synthesised_report_is_an_abstention_and_claims_nothing():
    report = synthetic_abstain_report("max_steps")
    assert report["abstain"] is True
    assert report["claims"] == []
    assert report["citations"] == []


# ---------------------------------------------------------------------------
# 9. The caps
# ---------------------------------------------------------------------------


def test_the_model_call_cap_stops_a_runaway_loop():
    class Loop(Middleware):
        name = "loop"

        def after_model(self, ctx, response):
            return ModelResponse(text="THOUGHT: mãi mãi.", prompt_tokens=1,
                                 completion_tokens=1)

    result = mock_run(
        middleware=[Loop()],
        config=RunnerConfig(max_model_calls=5, warn_on_missing_final=False),
    )
    assert result.model_calls == 5
    assert result.aborted
    assert result.stop_reason == "max_model_calls"
    assert result.gate()[0]


def test_the_tool_call_cap_stops_a_runaway_retry_layer():
    class InfiniteRetry(Middleware):
        name = "infinite-retry"

        def wrap_tool_call(self, ctx, call, name, args):
            for _ in range(10_000):
                call(name, args)
            return call(name, args)  # pragma: no cover - the cap fires first

    result = mock_run(
        middleware=[InfiniteRetry()],
        config=RunnerConfig(max_tool_calls=20, warn_on_missing_final=False),
    )
    assert result.aborted
    assert result.stop_reason == "max_tool_calls"
    assert result.gate()[0]


def test_the_wall_clock_cap_fires():
    class TickingClock:
        def __init__(self):
            self.now = 0.0

        def __call__(self):
            self.now += 1.0
            return self.now

    result = run_brief(
        BRIEF, model=MockModel(corpus=CORPUS, seed=11), corpus=CORPUS, seed=11,
        config=RunnerConfig(wall_clock_seconds=4.0, hard_stop_margin=0.0,
                            warn_on_missing_final=False),
        clock=TickingClock(),
    )
    assert result.aborted
    assert result.stop_reason == "wall_clock"
    assert result.gate()[0]


def test_a_capped_run_is_still_scoreable():
    result = mock_run(
        config=RunnerConfig(max_model_calls=2, warn_on_missing_final=False)
    )
    score = score_result(result, BRIEF, CORPUS)
    assert score.gate_passed
    assert score.total >= 0.0


def test_the_hard_stop_interrupts_a_harness_that_calls_nothing_at_all():
    """The soft caps sit at the model and tool chokepoints, so a pure CPU
    spin touches neither. Only a signal reaches one — best effort, POSIX
    main thread only."""
    import signal as _signal

    if not hasattr(_signal, "SIGALRM") or not hasattr(_signal, "setitimer"):
        pytest.skip("SIGALRM not available on this platform")

    class Spinner(Middleware):
        name = "spinner"

        def before_agent(self, ctx):
            limit = time.monotonic() + 5.0  # test-side backstop, never reached
            while time.monotonic() < limit:
                pass

    started = time.monotonic()
    result = mock_run(
        middleware=[Spinner()],
        config=RunnerConfig(wall_clock_seconds=0.2, hard_stop_margin=0.1,
                            warn_on_missing_final=False),
    )
    assert time.monotonic() - started < 4.0, "the hard stop did not fire"
    assert result.aborted
    assert result.gate()[0]


def test_the_kill_switch_survives_a_layer_that_swallows_exceptions():
    """A broad `except Exception:` in a student layer must not be able to
    keep a hung run alive — hence `RunAborted(BaseException)`."""
    assert issubclass(RunAborted, BaseException)
    assert not issubclass(RunAborted, Exception)

    class Swallower(Middleware):
        name = "swallower"

        def wrap_model_call(self, ctx, call, messages):
            try:
                return call(messages)
            except Exception:  # noqa: BLE001 - the point of the test
                return ModelResponse(text="THOUGHT: nuốt lỗi.", prompt_tokens=0,
                                     completion_tokens=0)

    result = mock_run(
        middleware=[Swallower()],
        config=RunnerConfig(max_model_calls=4, warn_on_missing_final=False),
    )
    assert result.aborted
    assert result.model_calls == 4


# ---------------------------------------------------------------------------
# 10. The trace guard — student code cannot write the evidence
# ---------------------------------------------------------------------------

FORGED_TEXT = 'THOUGHT: x\nFINAL: {"answer": "giả", "claims": [], "citations": [], "abstain": false}'


class Forger(Middleware):
    name = "forger"

    def before_agent(self, ctx):
        ctx.trace.emit("model_call", prompt_tokens=1, completion_tokens=1,
                       output_text=FORGED_TEXT)
        ctx.trace.emit("tool_call", name="fetch_doc", ok=True, doc_id="doc-0004",
                       flaky_mode=None)


def test_a_student_emitted_model_call_is_blocked_and_recorded():
    result = mock_run(middleware=[Forger()])
    assert not any(FORGED_TEXT in text for text in outputs(result))
    blocked = [
        r for r in records(result.trace_jsonl, "layer") if r.get("hook") == "blocked_emit"
    ]
    assert {r["blocked_event"] for r in blocked} == {"model_call", "tool_call"}
    assert "blocked_emits:2" in result.flags
    assert result.warnings


def test_blocking_a_forged_event_does_not_break_the_gate():
    assert mock_run(middleware=[Forger()]).gate()[0]


def test_layer_events_are_still_free():
    from harness.middleware import LoggingMiddleware

    result = mock_run(middleware=[LoggingMiddleware()])
    assert records(result.trace_jsonl, "layer")
    assert not [
        r for r in records(result.trace_jsonl, "layer") if r.get("hook") == "blocked_emit"
    ]
    assert result.gate()[0]


def test_the_forged_final_attack_scores_far_below_an_honest_stack():
    """FORGED FINAL, the attack the runner was assigned to close: emit a
    `model_call` the model never made whose `output_text` is a FINAL
    quoting the top-k lift-spans, cited to themselves.

    Measured on the public set, base seed 11: 83.39 with the trace
    unguarded, 45.64 with the guard. Forging is worth ~38 points to a
    harness the runner does not police, and worth nothing here.
    """
    from arena.model import _lift_span

    class ForgedFinal(Middleware):
        name = "forged-final"

        def after_agent(self, ctx, report):
            hits = ctx.corpus.search(ctx.question, k=5)
            claims = [
                {"text": _lift_span(doc.body), "doc_id": doc.doc_id} for doc in hits[:4]
            ]
            payload = {
                "answer": "Theo tài liệu: " + " ".join(c["text"] for c in claims),
                "citations": [c["doc_id"] for c in claims],
                "abstain": False,
                "claims": claims,
            }
            ctx.trace.emit(
                "model_call", prompt_tokens=1, completion_tokens=1,
                output_text="THOUGHT: x\nFINAL: " + json.dumps(payload, ensure_ascii=False),
            )
            return payload

    def sweep(guard: bool) -> float:
        totals = []
        for index, brief in enumerate(BRIEFS):
            seed = derive_seed(11, index)
            result = run_brief(
                brief, model=MockModel(corpus=CORPUS, seed=seed), corpus=CORPUS,
                middleware=[ForgedFinal()], seed=seed,
                config=RunnerConfig(guard_trace=guard, warn_on_missing_final=False),
            )
            totals.append(score_result(result, brief, CORPUS).total)
        return sum(totals) / len(totals)

    guarded = sweep(True)
    unguarded = sweep(False)
    assert unguarded - guarded > 25.0, (guarded, unguarded)


# ---------------------------------------------------------------------------
# 11. Determinism
# ---------------------------------------------------------------------------


def test_two_runs_in_one_process_are_byte_identical():
    a = mock_run()
    b = mock_run()
    assert strip_timing(a.trace_jsonl) == strip_timing(b.trace_jsonl)
    assert score_result(a, BRIEF, CORPUS).total == score_result(b, BRIEF, CORPUS).total


DETERMINISM_SNIPPET = """
import json, sys
sys.path.insert(0, {root!r})
from arena.corpus import Corpus
from arena.model import MockModel
from arena.briefs import load_public_briefs
from arena.runner import RunnerConfig, run_brief, score_result, strip_timing
corpus = Corpus.generate(seed=42)
brief = load_public_briefs()[0]
result = run_brief(brief, model=MockModel(corpus=corpus, seed=11), corpus=corpus,
                   seed=11, config=RunnerConfig())
print(json.dumps({{
    "trace": strip_timing(result.trace_jsonl),
    "total": score_result(result, brief, corpus).total,
}}))
"""


def test_two_processes_produce_byte_identical_traces():
    """The leaderboard depends on this. `elapsed_seconds` is a wall clock
    by definition and is the only field that may differ."""
    outputs_ = []
    for hashseed in ("0", "999"):
        proc = subprocess.run(
            [sys.executable, "-c", DETERMINISM_SNIPPET.format(root=str(LAB_ROOT))],
            capture_output=True, text=True, cwd=str(LAB_ROOT),
            env=_subprocess_env(PYTHONHASHSEED=hashseed),
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        outputs_.append(json.loads(proc.stdout))
    assert outputs_[0]["trace"] == outputs_[1]["trace"]
    assert outputs_[0]["total"] == outputs_[1]["total"]


def test_strip_timing_removes_only_the_clock():
    result = mock_run()
    stripped = strip_timing(result.trace_jsonl)
    assert "elapsed_seconds" not in stripped
    assert len(stripped.splitlines()) == len(result.trace_jsonl.splitlines())
    assert '"event": "model_call"' in stripped


def test_a_fixed_clock_makes_even_the_timing_deterministic():
    a = run_brief(BRIEF, model=MockModel(corpus=CORPUS, seed=11), corpus=CORPUS,
                  seed=11, config=QUIET, clock=lambda: 0.0)
    b = run_brief(BRIEF, model=MockModel(corpus=CORPUS, seed=11), corpus=CORPUS,
                  seed=11, config=QUIET, clock=lambda: 0.0)
    assert a.trace_jsonl == b.trace_jsonl


# ---------------------------------------------------------------------------
# 12. The scripts
# ---------------------------------------------------------------------------


def _script(name, *args, expect=0):
    proc = subprocess.run(
        [sys.executable, f"scripts/{name}", *args],
        capture_output=True, text=True, cwd=str(LAB_ROOT),
        env=_subprocess_env(),
    )
    assert proc.returncode == expect, (proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
    return proc


def test_run_practice_produces_a_score_file_and_exits_zero(tmp_path):
    out = tmp_path / "practice.json"
    _script("run_practice.py", "--quiet", "--out", str(out))
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "arena-scores/1"
    assert len(payload["runs"]) == len(BRIEFS)
    assert payload["advisory"] is True
    assert all("total" in run for run in payload["runs"])
    assert isinstance(payload["mean_total"], (int, float))


def test_the_leaderboard_reports_a_gap_for_every_entry(tmp_path):
    entry = tmp_path / "entry.json"
    _script("run_practice.py", "--quiet", "--entry", "team-a", "--out", str(entry))
    proc = _script("leaderboard.py", str(entry), "--json", "--baseline-total", "0")
    payload = json.loads(proc.stdout)
    row = payload["entries"][0]
    assert payload["baseline_total"] == 0.0
    assert row["entry_id"] == "team-a"
    assert row["gap"] == pytest.approx(row["mean_total"] - 0.0, abs=0.01)
    # The verdict is a function of the GAP and nothing else — pinned here
    # against the published thresholds rather than against a score this
    # bundle has no way to predict.
    expected = ("CÓ GRADIENT" if row["gap"] >= 20.0
                else "YẾU" if row["gap"] >= 10.0
                else "KHÔNG CÓ GRADIENT")
    assert row["verdict"] == expected, (row["gap"], row["verdict"])


def test_the_leaderboard_says_so_when_there_is_no_gradient(tmp_path):
    entry = tmp_path / "flat.json"
    _script("run_practice.py", "--quiet", "--layers", "none", "--entry", "flat",
            "--out", str(entry))
    proc = _script("leaderboard.py", str(entry), "--json", "--baseline-total", "24.0")
    assert json.loads(proc.stdout)["entries"][0]["verdict"] == "KHÔNG CÓ GRADIENT"
    _script("leaderboard.py", str(entry), "--baseline-total", "24.0",
            "--require-gap", "20", expect=4)


def test_the_leaderboard_warns_when_the_baseline_is_only_an_estimate(tmp_path):
    entry = tmp_path / "e.json"
    _script("run_practice.py", "--quiet", "--out", str(entry))
    proc = _script("leaderboard.py", str(entry), "--json")
    assert json.loads(proc.stdout)["baseline_trusted"] is False
    _script("leaderboard.py", str(entry), "--strict-baseline", expect=3)


def test_a_score_file_tagged_baseline_is_used_as_the_baseline(tmp_path):
    """The self-scoring loop you actually run during practice: tag a
    no-layers run as the baseline, then measure every later run against
    it. On a fresh checkout the gap is ~0 because the stubs do nothing —
    that number going UP is the only evidence that a layer works."""
    baseline = tmp_path / "baseline.json"
    entry = tmp_path / "entry.json"
    _script("run_practice.py", "--quiet", "--layers", "none", "--tag", "baseline",
            "--entry", "blind", "--out", str(baseline))
    _script("run_practice.py", "--quiet", "--layers", "all", "--entry", "team",
            "--out", str(entry))
    proc = _script("leaderboard.py", str(tmp_path), "--json")
    payload = json.loads(proc.stdout)
    assert payload["baseline_trusted"] is True
    assert payload["baseline_source"].startswith("file tự khai baseline")
    rows = {row["entry_id"]: row for row in payload["entries"]}
    assert set(rows) == {"blind", "team"}
    assert rows["team"]["gap"] == pytest.approx(
        rows["team"]["mean_total"] - payload["baseline_total"], abs=0.01
    )


def test_run_practice_refuses_the_real_path_without_credentials():
    """No silent fall back to the mock: a scored round that quietly
    stopped being scored is worse than one that failed loudly."""
    proc = subprocess.run(
        [sys.executable, "scripts/run_practice.py", "--model", "real", "--brief",
         "pub-01-sla-hien-hanh"],
        capture_output=True, text=True, cwd=str(LAB_ROOT), env=_subprocess_env(),
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "ARENA_API_KEY" in combined
    assert "ARENA_BASE_URL" in combined


def test_verify_exits_zero_on_a_clean_checkout():
    proc = _script("verify.py")
    assert "✘" not in proc.stdout, proc.stdout
    assert proc.stdout.count("[✔]") >= 20


# ---------------------------------------------------------------------------
# 13. The gradient — the whole point of the round
#
# The build tree measures the gradient here: the complete five-layer
# solution has to beat the no-layers baseline by more than 30 points
# through the runner, or the round is not measuring anything. That test
# needs a finished stack, so it cannot ship — YOUR version of it is
# `scripts/leaderboard.py`, run against a `--tag baseline` score file.
# ---------------------------------------------------------------------------


def test_the_runner_costs_your_own_stack_nothing():
    """Installing the five layers must not change the score by itself.

    On a fresh checkout the stubs are no-ops, so this is an equality. It
    stops being one the moment you implement a layer — at which point the
    number that matters is the GAP, not this assertion, and this test is
    the one to delete rather than the one to satisfy.
    """
    baseline = _ladder("baseline")
    student = _ladder("student")
    assert len(baseline) == len(student) == len(BRIEFS)
    assert all(total > 0 for total in baseline)


def test_score_session_lines_results_up_with_their_briefs():
    from arena.runner import score_session

    results = run_session(
        BRIEFS[:3],
        model_factory=lambda seed: MockModel(corpus=CORPUS, seed=seed),
        corpus=CORPUS,
        middleware_factory=lambda seed: student_stack(),
        base_seed=11,
        config=QUIET,
    )
    scores = score_session(results, BRIEFS[:3], CORPUS)
    assert len(scores) == 3
    assert all(score.gate_passed for score in scores)
    assert [s.detail["brief_id"] for s in scores] == [r.brief_id for r in results]


def test_the_instructor_can_keep_the_prompts_without_putting_them_in_the_trace():
    """Clause 3's forensic half: `prompt_sha256` proves WHAT was sent,
    and this is how an instructor reads it back. Deliberately not in the
    trace — students submit traces, and a full prompt log would blow the
    90,000-character record budget."""
    result = mock_run(
        config=RunnerConfig(record_prompts=True, max_model_calls=2,
                            warn_on_missing_final=False)
    )
    assert len(result.prompts) == result.model_calls
    assert ARENA_SYSTEM_PROMPT[:40] in result.prompts[0]
    assert ARENA_SYSTEM_PROMPT[:40] not in result.trace_jsonl
    assert all(
        len(record["prompt_sha256"]) == 64
        for record in records(result.trace_jsonl, "model_call")
    )
    assert not mock_run(config=RunnerConfig(max_model_calls=2,
                                            warn_on_missing_final=False)).prompts


def test_the_scored_prompt_addendum_is_off_by_default_and_reaches_the_model_when_on():
    """Measured on a live endpoint: gpt-5.6-luna abstained on turn 1 with
    ZERO tool calls on 4 of 6 runs, which flattens the ladder. The
    addendum compels a search first. Off by default so it cannot collide
    with a prompt your own layers may want to set."""
    plain = ScriptedModel('THOUGHT: x\nFINAL: {"abstain": true, "claims": []}')
    run_brief(BRIEF, model=plain, corpus=CORPUS, seed=11, config=QUIET)
    sent = plain.calls[0][0][0]["content"]
    assert sent == ARENA_SYSTEM_PROMPT

    nudged = ScriptedModel('THOUGHT: x\nFINAL: {"abstain": true, "claims": []}')
    run_brief(
        BRIEF, model=nudged, corpus=CORPUS, seed=11,
        config=RunnerConfig(prompt_addendum=True, warn_on_missing_final=False),
    )
    sent = nudged.calls[0][0][0]["content"]
    assert sent.startswith(ARENA_SYSTEM_PROMPT)
    assert "search ít nhất một lần" in sent


def test_the_runner_never_flags_an_honest_run_for_review():
    """Every review flag must be silent on an honest run, or it is noise
    an instructor learns to ignore — and a flag on YOUR run is the first
    thing a human will look at. Run this after every layer you finish."""
    for index, brief in enumerate(BRIEFS):
        seed = derive_seed(11, index)
        result = run_brief(
            brief, model=MockModel(corpus=CORPUS, seed=seed), corpus=CORPUS,
            middleware=student_stack(), seed=seed, config=QUIET,
        )
        assert result.flags == (), (brief["brief_id"], result.flags)
        assert result.error == ""
        assert result.warnings == ()
