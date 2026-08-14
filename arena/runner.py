"""The FROZEN runner — the component that decides what the scorer believes.

Students may rewrite anything under `harness/`. They may not touch this
file, and they should read it: everything the frozen scorer treats as
EVIDENCE is written here, by instructor-owned code, on the way past.

    trace = Trace(...)                 # the record
    tools = Tools(corpus, trace, ...)  # frozen, emits its own tool_call
    model = ProvenanceModel(real_or_mock, trace, ...)   # <- THIS FILE
    agent = ReActAgent(model, tools, trace, middleware=<student layers>)
    report = agent.run(brief)          # student code, from here down

WHY A FROZEN RUNNER EXISTS AT ALL
=================================

`arena/scorer.py` credits a claim only if its text appears in the parsed
FINAL payload of some `model_call` event's `output_text`. That single
rule is what prices document SELECTION — the skill this lab exists to
teach — and it is worth ~56 points on the trap-spanning set. **Whoever
controls `output_text` controls the score.** `wrap_model_call` and
`after_model` are student-owned hooks that receive and return the model
response, so a trace stamped from what those hooks returned would prove
nothing: one line, `response.text = ""`, restored a measured 100.00
exploit. So the stamp happens here, at the client boundary, and nothing
downstream can move it.

THE THREE CLAUSES THIS MODULE ENFORCES
======================================

1. **`output_text` comes from the RAW client response.** It is captured
   inside `ProvenanceModel.complete`, which is the innermost callable in
   the `wrap_model_call` onion — before `after_model`, and *outside*
   whatever `wrap_model_call` chooses to return. A hook may rewrite the
   response it hands its caller; it cannot rewrite what was already
   written to the trace. Probe:
   `tests/test_runner.py::test_a_hostile_after_model_cannot_rewrite_output_text`.

2. **Once a `model_call` event exists, `output_text` is present and
   non-empty.** The scorer's provenance rule may not self-disable, so the
   runner must never emit the field empty — a model that returns an empty
   string still gets a `model_call` carrying the marker
   `EMPTY_OUTPUT_SENTINEL`, which is honest, non-empty and decodes as
   nothing. Probe: `test_output_text_is_never_emitted_empty`.

3. **The runner owns the message list and records what it actually
   sent.** Every `model_call` carries `prompt_sha256` (over the messages
   as handed to the client), `prompt_chars`, `n_messages`, and
   `unaccounted_chars` — how much of the prompt the runner cannot account
   for out of the system prompt, the brief, the tool observations and the
   model's own prior turns. That last number is the one an instructor
   reads to see whether quoted "evidence" was pasted in by the harness
   rather than retrieved. It is INFORMATIONAL and never scored. Probe:
   `test_prompt_sha256_records_what_was_actually_sent`.

NORMALISATION HAPPENS **BEFORE** THE STAMP, AND THAT PLACEMENT WAS
MEASURED
==================================================================

Real endpoints indent, fence, bold, lower-case and pretty-print. Fixing
that only inside the runner's own parser moved end-to-end coverage of
realistic output shapes from 56.8% to 60.0%; doing it *before*
`output_text` is written moved it to 76.8%. The reason is structural: the
scorer re-derives provenance from `output_text` itself, so a smarter
runner-side parser that stamps the raw text repairs nothing the scorer
sees. `normalise_output` therefore runs first, and the stamped text is
what both the agent and the scorer read.

`normalise_output` REPAIRS, it does not REPLACE: the model's own words
are preserved, fence lines are dropped, a marker line is rewritten to
canonical `FINAL:`/`ACTION:` with its payload re-serialised onto one
line, and a payload the frozen parser still cannot see is appended as a
canonical line *only* when the turn is otherwise unreadable. It can
therefore never turn a working ACTION turn into a FINAL, and it can never
introduce a payload the scorer would not have found in the raw text
anyway.

WHAT ELSE THE RUNNER OWNS, AND WHY EACH ONE IS HERE
===================================================

* **`max_tokens` = 3000, negotiated.** `arena/model.py` defaults to 1024,
  which is ~70% of a dense four-claim Vietnamese FINAL — the payload gets
  cut mid-string, stops being JSON, and grounding silently goes to zero.
  Some endpoints accept only `max_completion_tokens`; the runner tries
  the frozen client's own parameter first and falls back automatically,
  remembering the answer for the rest of the session.
* **`agent_end` with `elapsed_seconds`.** A wall clock inside the harness
  would make the trace non-deterministic and forgeable; the runner
  measures it outside the student's code. `Trace.validate` requires only
  that the FIRST event is `agent_start` and the LAST is `agent_end`, so
  the runner's own `agent_end` is appended after the agent's.
* **The report is read back off the `submit` event's `report_json`.** The
  object the agent's `run()` returns and the object it actually submitted
  can differ; the scorer cross-checks claims against the submitted one,
  so that is the one to score.
* **A synthesised abstention when there is no FINAL.** An honest run that
  hits the step cap submits nothing, and nothing scores 0.00 — below what
  a one-character troll used to get. The runner submits an explicit
  `{"abstain": true, ...}` instead, which is a true statement about that
  run and scores like one.
* **A per-run wall clock, model-call cap and tool-call cap.** Fourteen
  teams share one scored batch; one hung harness must not stall it.
* **`model_call` and `tool_call` may only be emitted by frozen code.**
  Everything else (`agent_start`, `layer`, `agent_end`) is free. A
  student-emitted `model_call` is the FORGED FINAL attack; a
  student-emitted `tool_call` forges retrieval. Both are downgraded to a
  `layer` event that records the attempt, so the run finishes and the
  forgery is inert and visible.

DETERMINISM
===========

Everything the runner writes is deterministic given `(brief, seed,
model, middleware)` EXCEPT `elapsed_seconds`, which is a wall clock by
definition. `strip_timing()` removes it, and two runs of the same input
in two different processes are byte-identical afterwards. Pass
`clock=lambda: 0.0` to make even the timing deterministic (the scripts do
this nowhere; the tests do it on purpose).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass, field, replace

from arena.corpus import Corpus
from arena.model import (
    ARENA_SYSTEM_PROMPT,
    ModelResponse,
    RealModel,
    RealModelError,
    parse_output,
)
from arena.scorer import MODEL_OUTPUT_FIELD, score_run
from arena.scorer import _final_payloads as _scorer_final_payloads
from arena.scorer import _norm as _scorer_norm
from arena.tools import Tools
from arena.trace import Trace

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Bumped whenever the runner changes what it WRITES. Stamped on
#: `agent_start` so a leaderboard can tell two batches apart.
RUNNER_VERSION = "1.0"

#: Output budget the runner asks the endpoint for, per call.
#:
#: `arena/model.py` hardcodes 1024 and passes no kwargs. A realistic
#: four-claim Vietnamese FINAL measures ~1,615 characters ≈ 690-735 real
#: tokens BEFORE any THOUGHT or reasoning tokens, i.e. about 70% of that
#: cap: the payload is cut mid-string, stops being JSON, `final_outputs`
#: drops to 0 and all 55 grounding points go with it — with the trace
#: gate still passing, so it looks like the student's fault. 3000 is 4x
#: the measured worst case.
DEFAULT_MAX_TOKENS = 3000

#: Parameter names for the output budget, in the order the runner tries
#: them. `max_tokens` first because that is the one the frozen
#: `RealModel` sends natively, so the ordinary path runs entirely through
#: frozen code; `max_completion_tokens` is the fallback for endpoints
#: that reject the older spelling.
MAX_TOKENS_PARAMS = ("max_tokens", "max_completion_tokens")

#: `output_text` is clamped to this before it is stamped. `Trace.emit`
#: reshapes any record over 90,000 characters, and a truncated FINAL
#: stops being decodable JSON — measured: 88,422 characters fine, 90,422
#: dead, silently. Ordinary output is three orders of magnitude below.
MAX_OUTPUT_TEXT_CHARS = 60_000

#: What the runner stamps when the model genuinely returned nothing.
#: Clause 2 says `output_text` is never emitted empty once a `model_call`
#: exists — because the scorer's provenance rule treats an absent field
#: as "no model output was recorded" and a student-emptied field is
#: indistinguishable from that. This marker is honest (it says the model
#: said nothing), non-empty, and decodes as no payload at all.
EMPTY_OUTPUT_SENTINEL = "[ARENA-EMPTY-MODEL-OUTPUT]"

#: Runner-side backstops. `MAX_STEPS` lives in student-owned code and
#: therefore cannot be the instructor's kill switch.
#:
#: These are OPS limits, not scoring budgets — deliberately independent of
#: the brief's `budget` block, which the scorer reads in coarse buckets
#: and which a slow endpoint must be free to overrun. A run that reaches
#: any of these has already lost the whole wall-clock component; the cap
#: is here so one hung harness cannot stall a fourteen-team batch.
DEFAULT_MAX_MODEL_CALLS = 40
DEFAULT_MAX_TOOL_CALLS = 60
DEFAULT_WALL_CLOCK_SECONDS = 180.0

#: The largest `k` a scored `search` may ask for. **NOT a taste setting —
#: it is the number that makes `UNRETRIEVED` reachable at all.**
#:
#: `arena/tools.py` is frozen and passes `k` straight through to
#: `Corpus.search`, and `arena/scorer.py` credits retrieval by REPLAYING
#: each `search` event at its recorded `k`. One `tools.search(question,
#: k=120)` therefore marked all 120 documents retrieved in a single
#: stamped call, so every citation in the run became eligible for
#: retrieval credit and the `UNRETRIEVED` verdict could not fire. That is
#: the cheapest attack the Codex review found, and it falsifies the
#: multi-hop design's central claim ("a dump cannot supply a fact that was
#: never retrieved") on its own.
#:
#: The clamp is here rather than in `arena/tools.py` because that file is
#: frozen, and here rather than in `harness/agent.py` (which already
#: clamps at 20) because that file is STUDENT-OWNED and a student deletes
#: it by rewriting their agent.
#:
#: **The VALUE is derived, not chosen.** A conforming brief's DEPTH
#: property says the supporting document is not in the question's top
#: `arena.scorer.MAX_SCORED_CLAIMS` (10) hits. Clamping at exactly that
#: number means one question-shaped search can never retrieve a
#: DEPTH-conforming supporting document, whatever `k` the harness asks
#: for — the clamp and the brief contract now say the same thing. Every
#: honest path in this lab asks for `k=5` (the mock, the oracle, the
#: reference budgets module, `harness/agent.py`'s default), so the clamp
#: costs an honest agent nothing; measured in
#: `tests/test_adversary.py::test_the_k_clamp_costs_an_honest_run_nothing`.
MAX_SEARCH_K = 10

#: Review-flag thresholds for the DUMP SIGNATURE. See `dump_signature`.
#: These pick out runs an instructor should LOOK AT. They never change a
#: score, and they are deliberately not tuned to `model_calls == 1`: the
#: measured adaptive dumper makes TWO calls (name the topic, then repeat
#: the documents), and a three-call spelling is a one-line change.
DUMP_MODEL_CALL_CEILING = 3
DUMP_DISTINCT_DOCS_FLOOR = 8

#: How many lines a pretty-printed payload may span before the runner
#: gives up trying to fold it onto one line.
MAX_PAYLOAD_LINES = 60

#: How many marker-looking lines one output may be probed for. Purely a
#: DoS bound: the marker pattern matches any line starting "Finally,",
#: and folding is quadratic in the window it searches, so a pathological
#: output could otherwise cost seconds. Real output has one or two.
MAX_MARKER_PROBES = 200

#: How many recovered payloads may be appended as canonical FINAL lines
#: when the turn is otherwise unreadable. The scorer probes at most 50
#: markers per output; this stays far below that.
MAX_APPENDED_FINALS = 2

#: How much text the payload RECOVERY pass scans, from the end.
#:
#: A DoS bound, measured: 20,000 lines of prose beginning "Finally," all
#: match the scorer's marker pattern (its colon is optional), and each of
#: its 50 probes slices and translates the whole remaining string —
#: 855 ms on a one-megabyte output. By the wire protocol the FINAL is the
#: last thing written, so scanning the tail loses nothing real; and the
#: field the scorer itself reads is clamped to `MAX_OUTPUT_TEXT_CHARS`
#: anyway, so this only bounds the runner's own extra pass.
MAX_RECOVERY_SCAN_CHARS = 100_000

#: Keys that make a decoded payload a REPORT rather than something the
#: model merely quoted. Same tuple as the scorer's `_REPORT_KEYS`.
REPORT_KEYS = ("answer", "claims", "abstain", "citations")

#: The report the runner submits on the student's behalf when the agent
#: produced no FINAL at all. It is a TRUE statement about that run: no
#: answer was reached. Scoring it as an abstention rather than as silence
#: is what stops an honest run that hit the step cap from landing below a
#: one-character troll.
SYNTHETIC_ABSTAIN_ANSWER = (
    "Không đủ căn cứ để trả lời: agent kết thúc mà không đưa ra FINAL hợp lệ, "
    "nên không có kết luận nào được đưa ra."
)

#: Optional addendum for the SCORED path, applied only when
#: `RunnerConfig.prompt_addendum` is on. It exists because a live test
#: measured gpt-5.6-luna abstaining on turn 1 with ZERO tool calls on 4
#: of 6 runs: zero tools -> zero claims -> the abstain floor -> a ladder
#: with no gradient. `ARENA_SYSTEM_PROMPT` is frozen and never compels a
#: search before abstaining, so the compulsion has to be added here.
#: Describes the schema in WORDS and never shows a template — a template
#: in the prompt is quotable, and a quoted template shadows the real
#: report.
SCORED_PROMPT_ADDENDUM = """
QUY TẮC BỔ SUNG (bắt buộc):
5. Bạn PHẢI gọi công cụ search ít nhất một lần trước khi kết luận. Tuyệt đối không
   được trả lời "không có thông tin" ở lượt đầu tiên khi chưa tìm kiếm.
6. Dòng FINAL phải nằm trên MỘT dòng duy nhất: chữ FINAL, dấu hai chấm, rồi toàn bộ
   đối tượng JSON trên cùng dòng đó. Không dùng khối mã (```), không in đậm, không
   xuống dòng giữa JSON.
7. Mỗi doc_id có đúng dạng doc- kèm bốn chữ số.
8. Tối đa 4 phần tử trong claims; mỗi phần tử là một câu trích nguyên văn, không quá
   400 ký tự, và phải nằm gọn trong MỘT dòng của tài liệu được trích.
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RunnerError(RuntimeError):
    """Anything the frozen runner refuses to do."""


class PreflightError(RunnerError):
    """The provenance contract was not satisfied. LOUD ON PURPOSE.

    One dropped `output_text` converts an entire cohort's claims to
    `NOT_FROM_MODEL` — a ~55-point wipe with `gate_passed=True` and no
    other visible symptom. There is no safe way to discover that after
    the round.
    """


class RunAborted(BaseException):
    """A runner-enforced cap fired: wall clock, model calls or tool calls.

    Deliberately a `BaseException` and not an `Exception`. Student layers
    are ordinary Python and a broad `except Exception:` in one of them
    would otherwise swallow the instructor's kill switch and keep the
    batch hung — which is the exact failure this cap exists to prevent.
    """


# ---------------------------------------------------------------------------
# Output normalisation — BEFORE the stamp. See the module docstring.
# ---------------------------------------------------------------------------

_INVISIBLES = "﻿​‌‍⁠"
_FENCE_LINE_RE = re.compile(r"^[ \t]*```[A-Za-z0-9_+-]*[ \t]*$")
#: `FINAL:` / `ACTION:` as a real endpoint actually writes them:
#: indented, quoted, bulleted, bolded, heading-prefixed, lower-cased,
#: with a space before the colon — or with no colon at all. Nothing is
#: rewritten unless the text after the marker DECODES as JSON, so
#: "Finally, the answer is 2 days" matches the marker and is then left
#: exactly as the model wrote it.
_MARKER_RE = re.compile(
    r"^[ \t>﻿]*(?:[*_#\-]{1,6}[ \t]*)?(final|action)[ \t]*[*_]{0,2}[ \t]*:?"
    r"[ \t]*[*_]{0,2}[ \t]*",
    re.IGNORECASE,
)
_ACTION_MARKER_RE = re.compile(
    r"^[ \t>﻿]*(?:[*_#\-]{1,6}[ \t]*)?action[ \t]*[*_]{0,2}[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
_SMART_QUOTES = str.maketrans(
    {
        "“": '"', "”": '"', "„": '"', "‟": '"',
        "‘": "'", "’": "'", "«": '"', "»": '"',
    }
)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _dumps(payload) -> str:
    """One-line, byte-stable JSON — the exact spelling `arena.model`
    uses, so a canonical mock turn survives normalisation untouched."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _recover_payloads(text: str) -> list:
    """FINAL payloads the SCORER's own recogniser can see in `text`.

    Bounded twice: nothing to do without a JSON opener anywhere, and at
    most `MAX_RECOVERY_SCAN_CHARS` from the end. Never raises — a
    recovery pass that blew up would take a whole batch with it.
    """
    if "{" not in text and "[" not in text:
        return []
    window = text if len(text) <= MAX_RECOVERY_SCAN_CHARS else text[-MAX_RECOVERY_SCAN_CHARS:]
    try:
        payloads = _scorer_final_payloads(window)
    except Exception:  # pragma: no cover - the frozen scorer does not raise
        return []
    return [p for p in payloads if isinstance(p, dict)]


def _decode_prefix(text: str):
    """Decode the JSON value `text` STARTS with. Returns (payload, end).

    `raw_decode` already tolerates pretty-printing and trailing prose.
    Two bounded repairs on top, for the two malformations real models
    actually emit: curly quotes and a trailing comma. Both are applied to
    the model's own text — nothing is invented.
    """
    decoder = json.JSONDecoder()
    for candidate in _repair_candidates(text):
        try:
            payload, end = decoder.raw_decode(candidate)
        except Exception:
            continue
        if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
            payload = payload[0]
        if isinstance(payload, dict):
            return payload, (end if candidate is text else len(candidate))
    return None, 0


def _repair_candidates(text: str) -> list[str]:
    attempts = [text]
    smart = text.translate(_SMART_QUOTES)
    if smart != text:
        attempts.append(smart)
    for candidate in list(attempts):
        fixed = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        if fixed != candidate:
            attempts.append(fixed)
    return attempts


def _fold_payload(tail: str, lines: list[str], index: int):
    """Fold a payload that may span several lines onto one.

    Returns `(payload, lines_consumed, leftover)`. Tries the marker's own
    line first, then progressively more following lines — the FIRST width
    that decodes wins, which is what makes `FINAL: {` / pretty JSON / `}`
    / `ACTION: {...}` fold to the FINAL only and leave the ACTION alone.

    A marker line whose payload cannot even START here is not folded at
    all: "Finally, the answer is 2 days" matches the marker pattern, and
    searching sixty lines after every such line is how a normaliser turns
    quadratic on ordinary prose.
    """
    head = tail.lstrip()
    if head:
        if head[0] not in "{[":
            return None, 1, ""
    else:
        following = lines[index + 1].lstrip() if index + 1 < len(lines) else ""
        if not following or following[0] not in "{[":
            return None, 1, ""
    limit = min(len(lines) - index, MAX_PAYLOAD_LINES)
    for width in range(1, limit + 1):
        chunk = "\n".join([tail] + lines[index + 1 : index + width])
        payload, end = _decode_prefix(chunk.lstrip())
        if payload is None:
            continue
        stripped = chunk.lstrip()
        leftover = stripped[end:].strip()
        return payload, width, leftover
    return None, 1, ""


def normalise_output(text: str) -> str:
    """Repair one model turn into the shape the FROZEN parser wants.

    Applied by `ProvenanceModel` BEFORE `output_text` is stamped — see
    the module docstring for why that placement is the whole point and
    not a style choice.

    REPAIRS, NEVER REPLACES. Line order and every word the model wrote
    are preserved; what changes is:

      * a BOM or zero-width prefix is dropped;
      * CRLF becomes LF;
      * ```` ``` ```` fence lines are dropped (the fenced content stays);
      * a `FINAL`/`ACTION` marker line — indented, bulleted, bolded,
        heading-prefixed, lower-cased, with a space before the colon or
        with no colon — is rewritten as canonical `FINAL: <payload>` with
        the payload re-serialised onto that one line, curly quotes and a
        trailing comma repaired, and any prose after it kept on its own
        line;
      * and, ONLY if the result is still unreadable to `parse_output` and
        carries no ACTION marker, a payload the scorer's recogniser can
        still see is appended as a canonical `FINAL:` line.

    That last clause cannot manufacture provenance: the payloads it
    appends are exactly the ones `arena.scorer` would already have
    credited from the raw text. And because it only fires when the turn
    is otherwise unreadable, it can never end a run that was still
    working — an ACTION under a quoted report keeps winning.
    """
    if not isinstance(text, str):
        return ""
    body = text.replace("\r\n", "\n").replace("\r", "\n").lstrip(_INVISIBLES)
    lines = [line for line in body.split("\n") if _FENCE_LINE_RE.match(line) is None]

    out: list[str] = []
    index = 0
    probes = 0
    while index < len(lines):
        line = lines[index]
        match = _MARKER_RE.match(line) if probes < MAX_MARKER_PROBES else None
        if match is None:
            out.append(line)
            index += 1
            continue
        probes += 1
        payload, consumed, leftover = _fold_payload(
            line[match.end() :], lines, index
        )
        if payload is None:
            out.append(line)
            index += 1
            continue
        out.append(f"{match.group(1).upper()}: " + _dumps(payload))
        if leftover:
            out.append(leftover)
        index += consumed

    result = "\n".join(out)
    if parse_output(result).kind != "unparseable":
        return result
    if _ACTION_MARKER_RE.search(result):
        return result
    appended = 0
    for payload in _recover_payloads(result):
        if appended >= MAX_APPENDED_FINALS:
            break
        if not any(k in payload for k in REPORT_KEYS):
            continue
        result += "\nFINAL: " + _dumps(payload)
        appended += 1
    return result


def clamp_output_text(text: str, limit: int = MAX_OUTPUT_TEXT_CHARS) -> str:
    """Bound `output_text` while KEEPING the payload that carries the score.

    A naive head-clamp cuts the FINAL in half, which stops it being JSON
    and costs all 55 grounding points silently. So the canonical FINAL
    lines are extracted first and reserved; the prose is what gets cut.
    """
    if len(text) <= limit:
        return text
    finals = [
        "FINAL: " + _dumps(payload)
        for payload in _recover_payloads(text)[:MAX_APPENDED_FINALS]
    ]
    tail = "\n".join(finals)
    if not tail:
        return text[:limit]
    if len(tail) >= limit:
        return tail[:limit]
    head = text[: max(0, limit - len(tail) - 1)]
    return head + "\n" + tail


# ---------------------------------------------------------------------------
# The record: only frozen code may write `model_call` and `tool_call`
# ---------------------------------------------------------------------------

#: Events the runner reserves for itself and for `arena/tools.py`.
#: `agent_start`, `layer` and `agent_end` stay free — students are
#: encouraged to log.
GUARDED_EVENTS = ("model_call", "tool_call")


def _as_int_or_zero(value) -> int:
    """Coerce to int for the RECORD, never raising. A student may pass
    anything at all as `k`; the record still has to be JSON."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class _PermitToken:
    """The thing `_GuardedTrace` believes, minted fresh per window.

    The permit used to be the plain STRING `"tool_call"`, which made the
    guard a two-line bypass: `ctx.trace._arena_permit = "tool_call"` and
    then a hand-built frozen `Tools` writes whatever it likes onto the
    scored record — measured by an independent reviewer at 120 documents
    credited as retrieved through a clamp that caps a search at 10.

    A token is not a defence against a determined attacker in the same
    process (nothing in-process is — `arena.runner._PermitToken` is
    importable, and that residual is stated in `normalise_records`), but
    it does mean the guard can no longer be opened by ASSIGNING THE
    OBVIOUS STRING, and it gives the trace a reliable answer to "did
    frozen code write this record?" — which is what
    `normalise_records` needs.
    """

    __slots__ = ("event",)

    def __init__(self, event: str) -> None:
        self.event = event


class _GuardedTrace(Trace):
    """A `Trace` that refuses to let student code forge the evidence.

    `model_call` may only be written by `ProvenanceModel`; `tool_call`
    only by `_GuardedTools`. Anything else asking for one is DOWNGRADED
    to a `layer` event naming the attempt, rather than refused: a scored
    batch that dies because one team modified `harness/agent.py` is an
    ops failure, while a forged event that quietly counts is an integrity
    failure. Downgrading is neither — the run finishes, the forgery is
    inert, and the attempt is on the record.

    THREE THINGS HAPPEN HERE, AND ONLY THE FIRST IS OLD:

    1. **The permit.** A guarded event is only written while a
       `_PermitToken` for that event is open (see `_PermitToken`).
    2. **The `k` clamp, at the moment of writing.** Every `search` event
       is clamped to `max_search_k` as it goes onto the record, whoever
       asked. The clamp used to live only in `_GuardedTools.search`,
       which student code can step around by building its own frozen
       `Tools`; a clamp on the RECORD cannot be stepped around that way,
       because the scorer's retrieval replay reads the recorded `k`.
    3. **`normalise_records` at serialization.** The record handed to the
       scorer contains no guarded event that frozen code did not write,
       and no search wider than the clamp — whatever route wrote it.

    Nothing outside this module ever sees this class. A student running
    their own script with a plain `Trace` is unaffected.
    """

    def __init__(self, run_id: str, seed: int, max_search_k: int = MAX_SEARCH_K) -> None:
        super().__init__(run_id, seed)
        self._arena_permit = None
        self.blocked_emits: list[str] = []
        self.max_search_k = int(max_search_k)
        #: Indices into `self._events` written under a real permit —
        #: i.e. by `ProvenanceModel`, `_GuardedTools` or the runner's own
        #: synthesised submit. Everything else is inert.
        self._arena_owned: set[int] = set()
        #: How many searches were clamped ON THE WAY ONTO THE RECORD.
        #: `_GuardedTools` clamps before it emits, so this counter only
        #: ever moves for a harness that wrote the event some other way.
        self.emit_search_k_clamps = 0
        #: Set by `normalise_records`; read by `run_brief` for its flags.
        self.record_search_k_clamps = 0
        self.record_downgrades = 0

    # -- writing -------------------------------------------------------

    def _clamped_search_fields(self, fields: dict) -> dict:
        """`fields` with `k` capped, and the request preserved."""
        if fields.get("name") != "search" or "k" not in fields:
            return fields
        requested = fields.get("k")
        scored = clamp_search_k(requested, self.max_search_k)
        try:
            same = int(requested) == scored
        except (TypeError, ValueError):
            same = False
        if same:
            return fields
        self.emit_search_k_clamps += 1
        clamped = dict(fields)
        clamped["k"] = scored
        clamped["k_requested"] = _as_int_or_zero(requested)
        clamped["k_clamped_by"] = "arena.runner"
        return clamped

    def emit(self, event: str, **fields) -> None:
        permit = self._arena_permit
        owned = isinstance(permit, _PermitToken) and permit.event == event
        if event in GUARDED_EVENTS and not owned:
            self.blocked_emits.append(str(event))
            keys = ",".join(sorted(str(k) for k in fields))[:200]
            super().emit(
                "layer",
                layer="arena.runner",
                hook="blocked_emit",
                blocked_event=str(event),
                blocked_fields=keys,
                n_fields=len(fields),
            )
            return
        if event == "tool_call":
            fields = self._clamped_search_fields(fields)
        index = len(self._events)
        super().emit(event, **fields)
        if owned and len(self._events) > index:
            self._arena_owned.add(index)

    # -- reading -------------------------------------------------------

    def raw_guarded_counts(self) -> dict:
        """Guarded events on the record BEFORE normalisation.

        `run_brief` reconciles against this rather than against the
        serialized record, so normalising a forged event into a `layer`
        does not hide the forgery from the review flag. Making a run look
        clean by making its evidence inert would be the wrong trade.
        """
        counts = {"model_call": 0, "tool_call": 0}
        for record in self._events:
            event = record.get("event")
            if event in counts:
                counts[event] += 1
        return counts

    def normalise_records(self) -> tuple:
        """`(records, stats)` — the record as the scorer should read it.

        TWO normalisations, both of them refusals to CREDIT rather than
        punishments:

        * a guarded event no frozen object wrote becomes a `layer` event.
          It stays on the record, named, with its fields' keys; it simply
          stops being evidence. This is the same policy `emit` already
          applies to the naive route, extended to the routes that walk
          past `emit` entirely (`Trace.emit(ctx.trace, ...)` unbound, or
          appending to `_events` directly).
        * a `search` event asking for more than `max_search_k` is clamped
          to it. The scorer's retrieval replay reads `k` off the record,
          so this is the only place a clamp is actually binding.

        **THE RESIDUAL, STATED PLAINLY:** student code runs in this
        process and can import `arena.runner._PermitToken`, so ownership
        is not unforgeable and this module does not claim it is. What the
        clamp gives is different in kind: it is unconditional, it needs
        no counter to agree with it, and no forged token raises it. What
        ownership gives is that the cheap routes — the two-line ones an
        exploit spreads as — credit nothing. The backstop for the rest is
        the review flag plus instructor re-execution, which is where it
        has always been.

        Never raises, never mutates `self._events`, and is idempotent:
        two calls produce byte-identical output.
        """
        records = []
        downgrades = 0
        clamps = 0
        for index, record in enumerate(self._events):
            event = record.get("event")
            if event in GUARDED_EVENTS and index not in self._arena_owned:
                downgrades += 1
                keys = ",".join(sorted(str(k) for k in record if k not in _RECORD_KEYS))
                records.append(
                    {
                        "seq": record.get("seq", index),
                        "event": "layer",
                        "run_id": record.get("run_id", self.run_id),
                        "seed": record.get("seed", self.seed),
                        "layer": "arena.runner",
                        "hook": "unowned_event",
                        "unowned_event": str(event),
                        "unowned_fields": keys[:200],
                    }
                )
                continue
            if event == "tool_call" and record.get("name") == "search":
                requested = record.get("k")
                scored = clamp_search_k(requested, self.max_search_k)
                try:
                    same = int(requested) == scored
                except (TypeError, ValueError):
                    same = requested is None
                if not same:
                    clamps += 1
                    record = dict(record)
                    record["k"] = scored
                    record["k_requested"] = _as_int_or_zero(requested)
                    record["k_clamped_by"] = "arena.runner"
            records.append(record)
        self.record_search_k_clamps = clamps
        self.record_downgrades = downgrades
        return records, {"downgrades": downgrades, "search_k_clamps": clamps}

    def to_jsonl(self) -> str:
        records, _stats = self.normalise_records()
        return "\n".join(json.dumps(record, sort_keys=True) for record in records)


#: Reserved keys `normalise_records` does not repeat into the downgraded
#: event's field list.
_RECORD_KEYS = frozenset({"seq", "event", "run_id", "seed"})


class _permit:
    """Open the window in which one guarded event may be written."""

    __slots__ = ("trace", "event", "previous", "token")

    def __init__(self, trace, event: str) -> None:
        self.trace = trace
        self.event = event
        self.previous = None
        self.token = None

    def __enter__(self):
        if isinstance(self.trace, _GuardedTrace):
            self.previous = self.trace._arena_permit
            self.token = _PermitToken(self.event)
            self.trace._arena_permit = self.token
        return self.trace

    def __exit__(self, *exc):
        if isinstance(self.trace, _GuardedTrace):
            self.trace._arena_permit = self.previous
        return False


# ---------------------------------------------------------------------------
# The caps
# ---------------------------------------------------------------------------


class _Guard:
    """Wall clock, model calls and tool calls, checked at frozen chokepoints.

    `MAX_STEPS` lives in `harness/agent.py`, which students own, so it
    cannot be the instructor's backstop. These three can be, because
    every model call goes through `ProvenanceModel` and every tool call
    through `_GuardedTools`.
    """

    def __init__(self, *, deadline, max_model_calls, max_tool_calls, clock):
        self.deadline = deadline
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls
        self.clock = clock
        self.model_calls = 0
        self.tool_calls = 0
        self.reason = ""

    def _deadline_check(self) -> None:
        if self.deadline is not None and self.clock() >= self.deadline:
            self.reason = "wall_clock"
            raise RunAborted("wall-clock budget exhausted")

    def before_model_call(self) -> None:
        self._deadline_check()
        if self.max_model_calls is not None and self.model_calls >= self.max_model_calls:
            self.reason = "max_model_calls"
            raise RunAborted(f"model-call cap reached ({self.max_model_calls})")
        self.model_calls += 1

    def before_tool_call(self) -> None:
        self._deadline_check()
        if self.max_tool_calls is not None and self.tool_calls >= self.max_tool_calls:
            self.reason = "max_tool_calls"
            raise RunAborted(f"tool-call cap reached ({self.max_tool_calls})")
        self.tool_calls += 1


class _HardStop:
    """Best-effort SIGALRM backstop for a harness that spins without
    calling anything at all.

    The soft caps above catch every loop that touches the model or the
    tools, which is every realistic hang. A pure CPU spin touches
    neither, and only a signal can interrupt one. POSIX + main thread
    only; everywhere else this is a no-op and the soft caps stand alone.
    """

    def __init__(self, seconds) -> None:
        self.seconds = seconds
        self.armed = False
        self._previous = None

    def _usable(self) -> bool:
        return (
            self.seconds is not None
            and self.seconds > 0
            and hasattr(signal, "SIGALRM")
            and hasattr(signal, "setitimer")
            and threading.current_thread() is threading.main_thread()
        )

    def __enter__(self):
        if not self._usable():
            return self

        def _fire(signum, frame):  # pragma: no cover - timing dependent
            raise RunAborted("hard wall-clock stop")

        try:
            self._previous = signal.signal(signal.SIGALRM, _fire)
            signal.setitimer(signal.ITIMER_REAL, float(self.seconds))
            self.armed = True
        except Exception:  # pragma: no cover - platform dependent
            self.armed = False
        return self

    def __exit__(self, *exc):
        if self.armed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                if self._previous is not None:
                    signal.signal(signal.SIGALRM, self._previous)
            except Exception:  # pragma: no cover - platform dependent
                pass
        return False


# ---------------------------------------------------------------------------
# The tools the run actually gets
# ---------------------------------------------------------------------------


def clamp_search_k(k, limit: int = MAX_SEARCH_K) -> int:
    """The scored `k` for a requested one. Coerces, floors at 1, caps.

    Flooring at 1 rather than passing 0 through is deliberate: the
    scorer's replay reads `k = min(max(1, ...), n_docs)`, so a `k=0`
    search returns nothing to the harness while still crediting one
    document's retrieval. Aligning the tool with the replay removes that
    (small, free) mismatch.
    """
    try:
        requested = int(k)
    except (TypeError, ValueError):
        requested = 5
    return max(1, min(int(limit), requested))


class _GuardedTools(Tools):
    """`arena.tools.Tools` plus the runner's cap, clamp and emit permit.

    A subclass, not a proxy, so `isinstance(tools, Tools)` and every
    attribute a layer reads (`calls`, `_corpus`) keep working. Each of
    the four entry points checks the cap, opens the `tool_call` permit,
    and records what came back so the runner can account for the prompt.

    THIS CLASS IS ALSO THE RETRIEVAL-BREADTH METER. Counting tool CALLS
    was never enough: one call can exfiltrate the corpus. So it also
    counts, per run:

    * `distinct_docs_returned` — how many DIFFERENT documents the run's
      searches and fetches put in front of it. This is computed the same
      way the scorer computes `retrieved` (replaying the search through
      `Corpus.search` at the recorded `k`), so it is exactly the breadth
      the score is based on and not an estimate of it.
    * `bytes_returned` — the size of everything handed back.
    * `searches_clamped` / `max_search_k_requested` — whether a harness
      asked for more breadth than a scored run may have.

    All four are INFORMATIONAL. Nothing here changes a score.
    """

    def __init__(
        self, corpus, trace, seed, flaky, guard, sink, max_search_k=MAX_SEARCH_K
    ) -> None:
        super().__init__(corpus, trace, seed=seed, flaky=flaky)
        self._guard = guard
        self._sink = sink
        self.max_search_k = int(max_search_k)
        self.searches_clamped = 0
        self.max_search_k_requested = 0
        self.docs_returned: set[str] = set()
        self.bytes_returned = 0
        #: Bodies the run actually FETCHED, in order. The dump signature
        #: asks whether the submitted claims are a subset of these.
        self.fetched_bodies: list[str] = []

    @property
    def distinct_docs_returned(self) -> int:
        return len(self.docs_returned)

    def _guarded(self, method, *args, **kwargs):
        self._guard.before_tool_call()
        with _permit(self._trace, "tool_call"):
            result = method(*args, **kwargs)
        self._sink.record_observation(result)
        self.bytes_returned += len(getattr(result, "content", "") or "")
        return result

    def search(self, query: str, k: int = 5):
        try:
            requested = int(k)
        except (TypeError, ValueError):
            requested = 5
        self.max_search_k_requested = max(self.max_search_k_requested, requested)
        scored_k = clamp_search_k(k, self.max_search_k)
        if scored_k != requested:
            self.searches_clamped += 1
        result = self._guarded(super().search, query, k=scored_k)
        # The same replay the scorer performs, so the meter and the score
        # agree on what "retrieved" means even when the observation came
        # back degraded (the scorer replays search events regardless of
        # `ok`, and so must this).
        if isinstance(query, str) and query:
            for doc in self._corpus.search(query, k=scored_k):
                self.docs_returned.add(doc.doc_id)
        return result

    def fetch_doc(self, doc_id: str):
        result = self._guarded(super().fetch_doc, doc_id)
        if getattr(result, "ok", False):
            self.docs_returned.add(doc_id)
            self.fetched_bodies.append(getattr(result, "content", "") or "")
        return result

    def calc(self, expression: str):
        return self._guarded(super().calc, expression)

    def submit(self, report: dict):
        return self._guarded(super().submit, report)


# ---------------------------------------------------------------------------
# The corpus student code is allowed to hold
# ---------------------------------------------------------------------------


#: Marks a `Corpus` that has already been through `shield_corpus`, so
#: shielding is idempotent and cheap.
SHIELD_ATTR = "_arena_shielded"


def shield_corpus(corpus):
    """The corpus handed to STUDENT code, with `Doc.tags` stripped.

    `harness/agent.py` holds `tools._corpus` and publishes it as
    `ctx.corpus`, so `ctx.corpus.get(doc_id).tags` used to name every
    trap class — `('outdated', 'shipping_sla', 'operations')` — from
    inside a student layer. That is a free oracle for all three evidence
    layers at once: `citation_checker` can reject any document tagged
    `outdated`, `critic` can find the `contradiction` pair without
    reading a word, and `injection_guard` can skip the `injection` doc
    without ever seeing the canary. The three layers the round is meant
    to grade become table lookups.

    **STATED PLAINLY, BECAUSE IT MATTERS MORE THAN THE FIX: on the
    PUBLISHED practice corpus this buys nothing.** `data/corpus/*.json`
    ships `tags` on disk and `Corpus.generate(seed=42)` is a public pure
    function, so a practice-round student can hard-code the tag table in
    five minutes and no runtime shield can stop them. Assume they will.

    What the shield IS load-bearing for is the SCORED round, and only
    because the design rotates the corpus seed per scoring wave: on an
    unpublished seed there is no `data/corpus` to read and no seed to
    regenerate from, so the runtime handle is the ONLY tag oracle that
    exists — and this closes it. A shield without seed rotation is
    theatre; seed rotation without a shield is not rotation at all.

    Ranking is provably unaffected: `Corpus.__init__` tokenises
    `title + " " + body` and never reads `tags`, so the shielded corpus
    returns byte-identical search results. The instructor keeps the real
    corpus and scores against it — `score_run` is called with the
    unshielded object.
    """
    if corpus is None or getattr(corpus, SHIELD_ATTR, False):
        return corpus
    cached = getattr(corpus, "_arena_shield_cache", None)
    if cached is not None:
        return cached
    docs = getattr(corpus, "docs", None)
    if not isinstance(docs, list):
        return corpus
    shielded = Corpus([replace(doc, tags=()) for doc in docs])
    setattr(shielded, SHIELD_ATTR, True)
    try:
        corpus._arena_shield_cache = shielded
    except Exception:  # pragma: no cover - defensive; Corpus is a plain object
        pass
    return shielded


# ---------------------------------------------------------------------------
# The dump signature — a REVIEW FLAG, never a score
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DumpSignature:
    """Does this run look like "retrieve broadly, then quote it back"?

    **This never changes a score and must never be wired to one.** It is
    a triage list: at 100 entrants a leaked five-line dumper spreads
    through the cohort in minutes, and an instructor needs to know which
    submissions to read, not an automatic verdict on them.

    The three components are the ones the two reviews named, and the
    conjunction is deliberately weak on purpose:

    * `few_model_calls` — a dump decides nothing, so it needs one call to
      launder the text and at most one more to name a topic. An honest
      run on a DEPTH-conforming brief needs a turn to search, a turn to
      re-query, a turn to fetch and a turn to conclude.
    * `claims_from_fetched` — every scored claim is a substring of a body
      this run fetched. NECESSARY for a dump and true of honest work too
      (the scorer requires verbatim containment), so it carries almost no
      evidence by itself; it is here to exclude runs that are doing
      something else entirely.
    * `wide_retrieval` — distinct documents PUT IN FRONT OF the harness,
      counted the way the scorer counts `retrieved`.

    **Fable's caveat, restated so nobody forgets it:** a two-call adaptive
    dumper already evades any rule tuned to `model_calls == 1`, and
    padding the run with three junk model calls evades this one for the
    price of one line. It is a flag for the cheap copy-paste exploit, not
    a defence. The defence is the brief.
    """

    model_calls: int
    distinct_docs_returned: int
    bytes_returned: int
    n_claims: int
    n_claims_from_fetched: int
    few_model_calls: bool
    claims_from_fetched: bool
    wide_retrieval: bool
    #: Claims quoting text this run never fetched. MEASURED EVASION, on
    #: the record rather than in a comment: the offline-mining attacker
    #: (`unbounded_k`) reads the public seed-42 corpus on disk and never
    #: calls `fetch_doc` at all, so `claims_from_fetched` is False for it
    #: and the flag above does not fire. The `k` clamp is what stops that
    #: attacker (it lands on the abstain floor), NOT this flag — but the
    #: number belongs on the record so an instructor triaging 100 runs can
    #: apply a rule this module did not anticipate.
    n_claims_never_fetched: int = 0

    @property
    def flagged(self) -> bool:
        return self.few_model_calls and self.claims_from_fetched and self.wide_retrieval

    def as_dict(self) -> dict:
        payload = {
            field: getattr(self, field)
            for field in (
                "model_calls",
                "distinct_docs_returned",
                "bytes_returned",
                "n_claims",
                "n_claims_from_fetched",
                "n_claims_never_fetched",
                "few_model_calls",
                "claims_from_fetched",
                "wide_retrieval",
            )
        }
        payload["flagged"] = self.flagged
        return payload


def dump_signature(
    report,
    fetched_bodies,
    *,
    model_calls: int,
    distinct_docs_returned: int,
    bytes_returned: int = 0,
    model_call_ceiling: int = DUMP_MODEL_CALL_CEILING,
    distinct_docs_floor: int = DUMP_DISTINCT_DOCS_FLOOR,
) -> DumpSignature:
    """Compute the review flag. Pure, cheap, and grades nothing."""
    claims = report.get("claims") if isinstance(report, dict) else None
    texts = []
    if isinstance(claims, list):
        for claim in claims:
            if isinstance(claim, dict):
                text = claim.get("text")
            else:
                text = claim
            if isinstance(text, str) and text.strip():
                texts.append(_scorer_norm(text))
    blob = _scorer_norm("\n".join(b for b in fetched_bodies if isinstance(b, str)))
    from_fetched = sum(1 for text in texts if text and text in blob)
    never_fetched = len(texts) - from_fetched
    return DumpSignature(
        model_calls=int(model_calls),
        distinct_docs_returned=int(distinct_docs_returned),
        bytes_returned=int(bytes_returned),
        n_claims=len(texts),
        n_claims_from_fetched=from_fetched,
        n_claims_never_fetched=never_fetched,
        few_model_calls=int(model_calls) <= int(model_call_ceiling),
        claims_from_fetched=bool(texts) and from_fetched == len(texts),
        wide_retrieval=int(distinct_docs_returned) >= int(distinct_docs_floor),
    )


# ---------------------------------------------------------------------------
# The provenance boundary
# ---------------------------------------------------------------------------


class ProvenanceModel:
    """The FROZEN client wrapper. Clause 1, 2 and 3 all live in here.

    It is the innermost callable of the `wrap_model_call` onion:

        student wrap_model_call
          -> harness.agent._call_model
            -> ProvenanceModel.complete   <- the trace is written HERE
              -> the real client

    Everything above this line may return whatever it likes. Nothing
    above this line can change what was already written.

    `emits_model_call = True` tells `harness/agent.py` not to stamp its
    own `model_call` — and `_GuardedTrace` enforces that rather than
    trusting it, so deleting the check from the student-owned agent
    changes nothing.
    """

    #: Read by `harness.agent.ReActAgent._call_model`.
    emits_model_call = True

    def __init__(
        self,
        inner,
        trace,
        *,
        guard,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_tokens_param: str = "auto",
        temperature: float = 0.0,
        record_prompts: bool = False,
        system_prompt: str = "",
        question: str = "",
    ) -> None:
        self.inner = inner
        self.trace = trace
        self.guard = guard
        self.max_tokens = int(max_tokens)
        self.temperature = temperature
        self.record_prompts = bool(record_prompts)
        self.calls = 0
        self.final_outputs = 0
        self.empty_outputs = 0
        self.prompts: list[str] = []
        self.outputs: list[str] = []
        self.param_used = ""
        self._param_order = self._param_plan(max_tokens_param)
        self._accepts_kwargs = _accepts_keyword_arguments(inner)
        # Everything the runner can account for in a prompt. Grows as the
        # run proceeds; see `unaccounted_chars`. `_known_blob` is cached
        # and `_accounted` memoises decided messages, so the accounting
        # stays linear over a long run instead of quadratic.
        self._known: list[str] = [system_prompt or "", question or ""]
        self._known_blob: str | None = None
        self._accounted: set = set()

    # -- prompt accounting --------------------------------------------

    @staticmethod
    def _param_plan(preference: str) -> list[str]:
        if preference in MAX_TOKENS_PARAMS:
            return [preference] + [p for p in MAX_TOKENS_PARAMS if p != preference]
        return list(MAX_TOKENS_PARAMS)

    def _remember(self, text) -> None:
        if isinstance(text, str) and text:
            self._known.append(text)
            self._known_blob = None

    def record_observation(self, result) -> None:
        """Called by `_GuardedTools` — what the tools actually returned."""
        self._remember(getattr(result, "content", ""))
        self._remember(getattr(result, "error", "") or "")

    def _unaccounted_chars(self, snapshot: list[dict]) -> int:
        """How much of this prompt the runner cannot explain.

        INFORMATIONAL, never scored. The stock harness sends the system
        prompt, the brief, the tool observations and the model's own
        prior turns — all of which the runner already has. What is left
        over is text the harness wrote, and a `budget_policy` nudge is
        ~100 characters while a pasted document is several thousand. It
        is the number an instructor reads to answer "was this evidence
        retrieved, or typed in?".

        Deterministic: a message once accounted for stays accounted for,
        because the known text only ever grows.
        """
        total = 0
        for message in snapshot:
            content = message.get("content", "")
            if not content or content in self._accounted:
                continue
            if self._known_blob is None:
                self._known_blob = "\n".join(self._known)
            if content in self._known_blob:
                self._accounted.add(content)
                continue
            total += len(content)
        return total

    # -- the call ------------------------------------------------------

    def complete(self, messages, **kw) -> ModelResponse:
        self.guard.before_model_call()
        snapshot = _snapshot(messages)
        blob = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        unaccounted = self._unaccounted_chars(snapshot)

        response = self._invoke(snapshot)

        # --- CLAUSE 1: the raw text, captured here and nowhere else.
        raw = getattr(response, "text", "")
        if not isinstance(raw, str):
            raw = "" if raw is None else str(raw)
        normalised = normalise_output(raw)
        # --- CLAUSE 2: never emitted empty.
        stamped = clamp_output_text(normalised) or EMPTY_OUTPUT_SENTINEL
        if not normalised:
            self.empty_outputs += 1

        prompt_tokens = _as_positive_int(getattr(response, "prompt_tokens", 0))
        completion_tokens = _as_positive_int(getattr(response, "completion_tokens", 0))

        with _permit(self.trace, "model_call"):
            self.trace.emit(
                "model_call",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                # `str` is immutable, so what is recorded here cannot be
                # rewritten by later code holding the same object.
                **{MODEL_OUTPUT_FIELD: stamped},
                # --- CLAUSE 3: what was ACTUALLY sent.
                prompt_sha256=hashlib.sha256(blob.encode("utf-8")).hexdigest(),
                prompt_chars=len(blob),
                n_messages=len(snapshot),
                unaccounted_chars=unaccounted,
                call_index=self.calls,
                max_tokens=self.max_tokens,
            )

        self.calls += 1
        self.outputs.append(raw)
        self._remember(raw)
        self._remember(normalised)
        if parse_output(stamped).kind == "final":
            self.final_outputs += 1
        if self.record_prompts:
            self.prompts.append(blob)

        # The agent sees the NORMALISED text, so what it acts on and what
        # the scorer credits are the same string. Tokens are the client's
        # own numbers, untouched.
        return ModelResponse(
            text=normalised,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # -- the output budget, negotiated --------------------------------

    def _invoke(self, messages) -> ModelResponse:
        """Ask the client for `max_tokens` output, whichever spelling it
        takes.

        The FIRST attempt goes through the frozen client's own
        `complete()` — no duplicated transport, no duplicated response
        parsing. Only when an endpoint rejects the parameter by name does
        the runner post the alternative spelling itself, reusing the same
        frozen `_post` (and therefore the same URL scheme allow-list and
        the same auth header). The working spelling is remembered, so the
        cost is one wasted call per session, not per turn.
        """
        if not self._accepts_kwargs:
            return self.inner.complete(messages)

        if self.param_used and self.param_used != "max_tokens":
            return self._post_with_param(messages, self.param_used)

        primary = self._param_order[0]
        if primary != "max_tokens" and isinstance(self.inner, RealModel):
            return self._post_with_param(messages, primary)

        try:
            response = self.inner.complete(
                messages, max_tokens=self.max_tokens, temperature=self.temperature
            )
        except RealModelError as exc:
            fallback = self._fallback_param(exc)
            if fallback is None:
                raise
            response = self._post_with_param(messages, fallback)
            return response
        self.param_used = self.param_used or "max_tokens"
        return response

    def _fallback_param(self, exc) -> str | None:
        if not isinstance(self.inner, RealModel):
            return None
        message = str(exc).lower()
        if not any(hint in message for hint in _PARAM_REJECTION_HINTS):
            return None
        for name in self._param_order:
            if name != "max_tokens":
                return name
        return None  # pragma: no cover - MAX_TOKENS_PARAMS always has two

    def _post_with_param(self, messages, param: str) -> ModelResponse:
        """One chat-completions POST with an explicit budget parameter.

        Mirrors `arena.model.RealModel.complete`'s payload and response
        handling deliberately and minimally: the frozen client always
        sends `max_tokens`, and an endpoint that accepts only
        `max_completion_tokens` cannot be reached any other way without
        editing a frozen file. Transport, scheme allow-list and auth all
        still come from `RealModel._post`.
        """
        payload = {
            "model": self.inner.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.temperature,
            param: self.max_tokens,
        }
        try:
            data = self.inner._post(payload)
        except RealModelError:
            raise
        except Exception as exc:
            raise RealModelError(
                f"Gọi endpoint thất bại ({self.inner.base_url}, tham số {param}): {exc}. "
                "Không có phương án dự phòng nào được dùng thay thế."
            ) from exc
        try:
            text = data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RealModelError(
                f"Endpoint {self.inner.base_url} trả về cấu trúc không hợp lệ "
                f"(thiếu choices[0].message.content): {exc}"
            ) from exc
        if not isinstance(text, str):
            raise RealModelError(
                f"Endpoint {self.inner.base_url} trả về content không phải chuỗi: "
                f"{type(text)!r}"
            )
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
            prompt_tokens = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
        if not isinstance(completion_tokens, int) or completion_tokens <= 0:
            completion_tokens = max(1, len(text) // 4)
        self.param_used = param
        return ModelResponse(
            text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )


#: Substrings an endpoint uses when it rejects the output-budget
#: parameter by name. Matched case-insensitively against the
#: `RealModelError` message, which carries the endpoint's own body.
_PARAM_REJECTION_HINTS = (
    "max_tokens",
    "max_completion_tokens",
    "unsupported_parameter",
    "unsupported parameter",
    "unrecognized",
    "unrecognised",
    "not supported",
)


def _accepts_keyword_arguments(model) -> bool:
    """Does this client's `complete` take `**kw`, or a `max_tokens`?

    Asked ONCE, by signature, rather than by catching `TypeError` around
    the call — a `TypeError` raised deep inside a model would otherwise
    be mistaken for a signature mismatch and the call silently retried.
    """
    try:
        signature = inspect.signature(model.complete)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return True
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name in ("max_tokens", "temperature"):
            return True
    return False


def _snapshot(messages) -> list[dict]:
    """A defensive copy of the message list as it was actually sent."""
    out: list[dict] = []
    if not isinstance(messages, (list, tuple)):
        return out
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role", "")
            content = message.get("content", "")
        else:  # pragma: no cover - a harness sending non-dicts
            role, content = "", message
        out.append(
            {
                "role": role if isinstance(role, str) else str(role),
                "content": content if isinstance(content, str) else str(content),
            }
        )
    return out


def _as_positive_int(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            value = int(value)
        except (TypeError, ValueError):
            return 0
    return max(0, value)


# ---------------------------------------------------------------------------
# Configuration and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunnerConfig:
    """Everything the runner will do, in one object.

    The defaults are the SCORED settings. Practice differs only in the
    model and the brief set.
    """

    max_tokens: int = DEFAULT_MAX_TOKENS
    max_tokens_param: str = "auto"
    temperature: float = 0.0
    max_model_calls: int | None = DEFAULT_MAX_MODEL_CALLS
    max_tool_calls: int | None = DEFAULT_MAX_TOOL_CALLS
    wall_clock_seconds: float | None = DEFAULT_WALL_CLOCK_SECONDS
    #: Best-effort SIGALRM stop for a harness that spins without calling
    #: anything. Adds a small margin over the soft deadline so the soft
    #: caps (which produce a scoreable run) always fire first.
    hard_stop_margin: float = 15.0
    flaky: bool = True
    max_steps: int | None = None
    #: Largest `k` a search may ask for. See `MAX_SEARCH_K`: this is the
    #: number that makes the scorer's `UNRETRIEVED` verdict reachable.
    max_search_k: int = MAX_SEARCH_K
    #: Hand student code a tag-stripped corpus. See `shield_corpus` —
    #: including the part about what it does NOT buy on a published seed.
    shield_corpus: bool = True
    system_prompt: str = ARENA_SYSTEM_PROMPT
    prompt_addendum: bool = False
    guard_trace: bool = True
    synthesise_abstain: bool = True
    record_prompts: bool = False
    #: Warn on stderr when a run records no decodable FINAL. The batch
    #: level check (`assert_provenance`) is the one that raises.
    warn_on_missing_final: bool = True

    def resolved_system_prompt(self) -> str:
        if not self.prompt_addendum:
            return self.system_prompt
        return self.system_prompt.rstrip() + "\n" + SCORED_PROMPT_ADDENDUM.strip()


@dataclass(frozen=True)
class RunResult:
    """One brief, run once. Everything a grader or an instructor needs."""

    brief_id: str
    seed: int
    run_id: str
    report: dict
    report_source: str
    trace_jsonl: str
    elapsed_seconds: float
    model_calls: int
    final_outputs: int
    tool_calls: int
    stop_reason: str
    aborted: bool = False
    error: str = ""
    warnings: tuple = ()
    flags: tuple = ()
    prompts: tuple = ()
    param_used: str = ""
    #: Retrieval BREADTH, not tool-call count. One call can exfiltrate the
    #: corpus, so the number of calls was never the interesting quantity.
    distinct_docs_returned: int = 0
    bytes_returned: int = 0
    searches_clamped: int = 0
    max_search_k_requested: int = 0
    #: INTEGRITY, on the record rather than in a comment. Guarded events
    #: the frozen runner did not write (downgraded to `layer`, crediting
    #: nothing) and searches clamped ON THE RECORD because they reached it
    #: without passing `_GuardedTools`. Both are zero on every honest run.
    record_downgrades: int = 0
    record_search_k_clamps: int = 0
    #: `DumpSignature.as_dict()`. INFORMATIONAL — see `dump_signature`.
    dump_signature: dict = field(default_factory=dict)

    @property
    def provenance_ok(self) -> bool:
        """Did at least one `model_call` record a decodable FINAL?

        The single question a preflight asks. `False` means every claim
        in this run will score `NOT_FROM_MODEL`.
        """
        return self.final_outputs > 0

    def gate(self) -> tuple:
        return Trace.validate(self.trace_jsonl)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def derive_seed(base_seed: int, index: int) -> int:
    """Per-brief seed. REQUIRED, and the reason is measured.

    `arena/tools.py` keys flakiness on `(seed, call index)`, so running a
    whole brief set on one seed replays the IDENTICAL failure pattern on
    every brief — at seed 42 the only failure lands on a padding call and
    flakiness effectively vanishes. `base + index` is also exactly the
    convention `tests/test_layers_reference.py` uses, so ladder numbers
    measured there reproduce here.
    """
    return int(base_seed) + int(index)


def _build_agent(model, tools, trace, middleware, corpus, config):
    """Construct the STUDENT's agent, passing only what it accepts.

    `harness/agent.py` is student-owned and may be rewritten. The runner
    depends on exactly one thing: `Agent(model, tools, trace,
    middleware=...)` with a `.run(brief) -> dict`.
    """
    from harness.agent import ReActAgent

    kwargs = {"middleware": middleware}
    try:
        parameters = inspect.signature(ReActAgent.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic classes
        parameters = {}
    if "corpus" in parameters:
        kwargs["corpus"] = corpus
    if "system_prompt" in parameters:
        kwargs["system_prompt"] = config.resolved_system_prompt()
    if config.max_steps is not None and "max_steps" in parameters:
        kwargs["max_steps"] = config.max_steps
    return ReActAgent(model, tools, trace, **kwargs)


def _submitted_report(jsonl: str):
    """The report the run ACTUALLY submitted, off the `submit` event.

    The object `run()` returns and the object it passed to
    `tools.submit()` can differ — a layer may return one thing and submit
    another, and the scorer cross-checks every claim against the
    submitted one (`NOT_SUBMITTED`). So the submitted one is the one to
    score. Returns `None` if there is no usable submit event.
    """
    found = None
    for line in jsonl.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if not isinstance(record, dict) or record.get("event") != "tool_call":
            continue
        if record.get("name") != "submit":
            continue
        payload = record.get("report_json")
        if not isinstance(payload, str) or not payload:
            continue
        try:
            decoded = json.loads(payload)
        except Exception:
            continue
        if isinstance(decoded, dict):
            found = decoded
    return found


def _is_submission(report) -> bool:
    """Did the run say ANYTHING a scorer can grade?

    Mirrors the frozen scorer's own "no submission" branch: an explicit
    abstention counts, a claim with real text counts, a real answer
    counts. Nothing else does.
    """
    if not isinstance(report, dict) or not report:
        return False
    if report.get("abstain") is True:
        return True
    claims = report.get("claims")
    if isinstance(claims, list):
        for claim in claims:
            if isinstance(claim, dict):
                text = claim.get("text")
                if isinstance(text, str) and text.strip():
                    return True
    answer = report.get("answer")
    return isinstance(answer, str) and bool(answer.strip())


def _guarded_event_counts(jsonl: str) -> dict:
    """How many `model_call` / `tool_call` events are ON THE TRACE.

    Compared against the runner's own counters, this is the whole
    forgery check: `ProvenanceModel` and `_GuardedTools` are the only
    frozen writers of those two events, and each increments a counter
    exactly once per event it writes. Any other number came from
    somewhere else.
    """
    counts = {"model_call": 0, "tool_call": 0}
    for line in jsonl.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        event = record.get("event") if isinstance(record, dict) else None
        if event in counts:
            counts[event] += 1
    return counts


def synthetic_abstain_report(reason: str) -> dict:
    """The report the runner submits when the agent produced no FINAL."""
    return {
        "answer": SYNTHETIC_ABSTAIN_ANSWER,
        "citations": [],
        "abstain": True,
        "claims": [],
        "arena_synthesised": reason,
    }


@dataclass(frozen=True)
class ScoredBoundary:
    """The four instructor-owned objects a scored run is made of.

    Exposed so that anything claiming to measure "what a student could
    submit" is measured at the SAME boundary a student submits through —
    same `k` clamp, same caps, same provenance stamp, same shielded
    corpus. `instructor/reference/adversary.py` builds its attackers on
    this for exactly that reason: an acceptance adversary that runs on a
    softer boundary than the students is a strawman, which is the defect
    that let a 97.28 attack through the previous acceptance harness.
    """

    trace: Trace
    guard: object
    client: "ProvenanceModel"
    tools: Tools
    corpus: Corpus


def scored_boundary(
    brief: dict,
    *,
    model,
    corpus: Corpus,
    seed: int = 42,
    run_id: str = "run",
    config: RunnerConfig | None = None,
    clock=time.perf_counter,
    started: float | None = None,
) -> ScoredBoundary:
    """Build the objects `run_brief` runs a student's agent on."""
    config = config or RunnerConfig()
    brief = brief if isinstance(brief, dict) else {}
    trace = (
        _GuardedTrace(run_id=run_id, seed=seed, max_search_k=config.max_search_k)
        if config.guard_trace
        else Trace(run_id=run_id, seed=seed)
    )
    started = clock() if started is None else started
    deadline = (
        started + config.wall_clock_seconds
        if config.wall_clock_seconds is not None
        else None
    )
    guard = _Guard(
        deadline=deadline,
        max_model_calls=config.max_model_calls,
        max_tool_calls=config.max_tool_calls,
        clock=clock,
    )
    client = ProvenanceModel(
        model,
        trace,
        guard=guard,
        max_tokens=config.max_tokens,
        max_tokens_param=config.max_tokens_param,
        temperature=config.temperature,
        record_prompts=config.record_prompts,
        system_prompt=config.resolved_system_prompt(),
        question=str(brief.get("question_vi", "")),
    )
    # Student code — the agent, its layers, and `ctx.corpus` — only ever
    # holds the SHIELDED corpus. The instructor keeps `corpus` and scores
    # against it.
    runtime_corpus = shield_corpus(corpus) if config.shield_corpus else corpus
    tools = _GuardedTools(
        corpus=runtime_corpus,
        trace=trace,
        seed=seed,
        flaky=config.flaky,
        guard=guard,
        sink=client,
        max_search_k=config.max_search_k,
    )
    return ScoredBoundary(
        trace=trace, guard=guard, client=client, tools=tools, corpus=runtime_corpus
    )


def run_brief(
    brief: dict,
    *,
    model,
    corpus: Corpus,
    middleware=None,
    seed: int = 42,
    run_id: str | None = None,
    config: RunnerConfig | None = None,
    clock=time.perf_counter,
) -> RunResult:
    """Run one brief end to end, under the frozen runner. Never raises.

    Returns a `RunResult` whose `trace_jsonl` always passes
    `Trace.validate` and whose `report` is the one recorded on the
    `submit` event. Student code failing is a RESULT (an error string and
    a zero-scoring report), not an exception — one team's bug must not
    take the batch down.
    """
    config = config or RunnerConfig()
    brief = brief if isinstance(brief, dict) else {}
    brief_id = str(brief.get("brief_id", ""))
    run_id = run_id or f"{brief_id or 'run'}-{seed}"

    started = clock()
    boundary = scored_boundary(
        brief,
        model=model,
        corpus=corpus,
        seed=seed,
        run_id=run_id,
        config=config,
        clock=clock,
        started=started,
    )
    trace, guard, client, tools = (
        boundary.trace,
        boundary.guard,
        boundary.client,
        boundary.tools,
    )
    runtime_corpus = boundary.corpus

    warnings: list[str] = []
    flags: list[str] = []
    error = ""
    aborted = False
    stop_reason = ""
    returned: dict = {}
    steps = 0

    # The runner owns BOTH ends of the trace. `Trace.validate` demands
    # that the first event is `agent_start` and the last is `agent_end`;
    # writing both here means the gate cannot fail because a student's
    # agent died before it wrote its own (the agent writes its own too,
    # and a second `agent_start` in the middle is legal).
    trace.emit(
        "agent_start",
        brief_id=brief_id,
        runner=RUNNER_VERSION,
        max_tokens=config.max_tokens,
        flaky=bool(config.flaky),
    )

    hard_seconds = (
        None
        if config.wall_clock_seconds is None
        else config.wall_clock_seconds + max(0.0, config.hard_stop_margin)
    )
    try:
        agent = _build_agent(client, tools, trace, middleware, runtime_corpus, config)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        agent = None

    if agent is not None:
        try:
            with _HardStop(hard_seconds):
                returned = agent.run(brief)
        except RunAborted as exc:
            aborted = True
            stop_reason = guard.reason or "aborted"
            error = f"RunAborted: {exc}"
        except BaseException as exc:  # noqa: BLE001 - one team must not stall the batch
            error = f"{type(exc).__name__}: {exc}"
        context = getattr(agent, "last_context", None)
        if context is not None:
            stop_reason = stop_reason or str(getattr(context, "stop_reason", ""))
            steps = int(getattr(context, "step", 0)) + 1

    if not isinstance(returned, dict):
        returned = {}

    # --- the report the run actually submitted --------------------------
    submitted = _submitted_report(trace.to_jsonl())
    if submitted is not None:
        report, report_source = submitted, "submit_event"
    elif returned:
        report, report_source = returned, "agent_return"
        warnings.append(
            "no usable submit event: scoring the object run() returned instead"
        )
        flags.append("no_submit_event")
    else:
        report, report_source = {}, "none"

    # --- an honest run that produced no FINAL is an ABSTENTION, not
    # --- silence. Silence scores 0.00; the abstention scores what an
    # --- abstention is worth. A run that DIED still scores zero.
    # A run the RUNNER stopped still did the work, so it gets the same
    # treatment. A run that DIED inside student code does not: "your
    # layer raises and you score zero" is the documented rule.
    if config.synthesise_abstain and not _is_submission(report) and (aborted or not error):
        report = synthetic_abstain_report(stop_reason or "no_final")
        report_source = "synthesised"
        with _permit(trace, "tool_call"):
            Tools.submit(tools, report)
        flags.append("synthesised_abstain")

    elapsed = max(0.0, float(clock() - started))
    trace.emit(
        "agent_end",
        stop_reason=stop_reason or ("error" if error else "unknown"),
        elapsed_seconds=round(elapsed, 6),
        runner=RUNNER_VERSION,
        model_calls=client.calls,
        final_outputs=client.final_outputs,
        aborted=bool(aborted),
        # Retrieval BREADTH, on the record. Never scored; the scorer
        # derives its own `retrieved` set by replay and does not read
        # these. They are what an instructor triages 100 runs with.
        distinct_docs_returned=tools.distinct_docs_returned,
        bytes_returned=tools.bytes_returned,
        searches_clamped=tools.searches_clamped,
        max_search_k_requested=tools.max_search_k_requested,
    )

    jsonl = trace.to_jsonl()
    blocked = list(getattr(trace, "blocked_emits", ()))
    if blocked:
        flags.append(f"blocked_emits:{len(blocked)}")
        warnings.append(
            "student code tried to write "
            + ",".join(sorted(set(blocked)))
            + " directly; the attempt was recorded and ignored"
        )
    if client.calls and client.final_outputs == 0:
        flags.append("no_final_output")
        if config.warn_on_missing_final:
            print(
                f"[arena.runner] CẢNH BÁO: brief {brief_id!r} (seed {seed}) không có "
                "model_call nào chứa FINAL đọc được. Mọi claim sẽ bị chấm "
                "NOT_FROM_MODEL.",
                file=sys.stderr,
            )
    if client.calls == 1:
        flags.append("single_model_call")

    # --- INTEGRITY: the runner knows both sides, so it compares them.
    # `_GuardedTrace` keys its refusal on an instance attribute, which
    # student code can set, unbind past (`Trace.emit(ctx.trace, ...)`) or
    # side-step entirely by building its own `ProvenanceModel` around a
    # fake client. Each route leaves a guarded event on the trace that no
    # frozen object produced, so counting is what catches all three.
    # A FLAG, never a failure: a false positive must not zero anyone.
    # Reconcile against the RAW record, not the normalised one: making a
    # forged event inert must not also make it invisible.
    counted = (
        trace.raw_guarded_counts()
        if isinstance(trace, _GuardedTrace)
        else _guarded_event_counts(jsonl)
    )
    if counted["model_call"] != client.calls or counted["tool_call"] != tools.calls:
        flags.append("review:event_reconciliation")
        warnings.append(
            "trace carries model_call/tool_call events the frozen runner did not "
            f"write (trace {counted['model_call']}/{counted['tool_call']}, runner "
            f"{client.calls}/{tools.calls}); read this run before trusting its score"
        )

    # --- INTEGRITY, second half: what the normalisation actually DID.
    # `record_downgrades` is guarded events that reached the record
    # without frozen code writing them (they are now inert `layer`
    # events); `record_search_k_clamps` is searches that reached the
    # record asking for more breadth than a scored run may have.
    # `_GuardedTools` clamps before it emits, so an honest agent asking
    # for `k=50` moves `searches_clamped` and NOT this counter.
    record_downgrades = int(getattr(trace, "record_downgrades", 0))
    record_search_k_clamps = int(
        getattr(trace, "record_search_k_clamps", 0)
    ) + int(getattr(trace, "emit_search_k_clamps", 0))
    if record_downgrades:
        flags.append(f"review:unowned_guarded_events:{record_downgrades}")
        warnings.append(
            f"{record_downgrades} guarded event(s) reached the trace without the "
            "frozen runner writing them; they were downgraded to `layer` and "
            "credit nothing"
        )
    if record_search_k_clamps:
        flags.append(f"review:search_k_bypass:{record_search_k_clamps}")
        warnings.append(
            f"{record_search_k_clamps} search event(s) reached the trace asking for "
            f"more than k={config.max_search_k}; the record was clamped"
        )

    signature = dump_signature(
        report,
        tools.fetched_bodies,
        model_calls=client.calls,
        distinct_docs_returned=tools.distinct_docs_returned,
        bytes_returned=tools.bytes_returned,
    )
    if signature.flagged:
        flags.append("review:dump_signature")

    if steps and client.calls < steps:
        flags.append("short_circuited_model_calls")
    if client.empty_outputs:
        flags.append(f"empty_model_output:{client.empty_outputs}")
    if aborted:
        flags.append(f"aborted:{guard.reason or 'unknown'}")
    elif error:
        flags.append("error")

    return RunResult(
        brief_id=brief_id,
        seed=seed,
        run_id=run_id,
        report=report,
        report_source=report_source,
        trace_jsonl=jsonl,
        elapsed_seconds=elapsed,
        model_calls=client.calls,
        final_outputs=client.final_outputs,
        tool_calls=tools.calls,
        stop_reason=stop_reason,
        aborted=aborted,
        error=error,
        warnings=tuple(warnings),
        flags=tuple(flags),
        prompts=tuple(client.prompts),
        param_used=client.param_used,
        distinct_docs_returned=tools.distinct_docs_returned,
        bytes_returned=tools.bytes_returned,
        searches_clamped=tools.searches_clamped,
        max_search_k_requested=tools.max_search_k_requested,
        record_downgrades=record_downgrades,
        record_search_k_clamps=record_search_k_clamps,
        dump_signature=signature.as_dict(),
    )


def run_session(
    briefs,
    *,
    model_factory,
    corpus: Corpus,
    middleware_factory=None,
    base_seed: int = 11,
    config: RunnerConfig | None = None,
    clock=time.perf_counter,
    on_result=None,
    strict_provenance: bool = False,
) -> list:
    """Run a whole brief set, one fresh model and stack per brief.

    `model_factory(seed)` and `middleware_factory(seed)` are called once
    per brief so no state leaks between runs — a layer that accumulated
    across briefs would make the leaderboard order-dependent.

    `strict_provenance=True` is the SCORED setting: assert after EVERY
    run that the run recorded a decodable FINAL, and stop the batch the
    moment one does not. It is deliberately not the default, because on
    the practice path a genuine "the model never finished" run is
    information rather than an emergency.
    """
    results = []
    for index, brief in enumerate(briefs):
        seed = derive_seed(base_seed, index)
        middleware = middleware_factory(seed) if middleware_factory else None
        result = run_brief(
            brief,
            model=model_factory(seed),
            corpus=corpus,
            middleware=middleware,
            seed=seed,
            config=config,
            clock=clock,
        )
        results.append(result)
        if on_result is not None:
            on_result(result)
        if strict_provenance:
            assert_provenance(
                [result], where=f"run {index + 1}/{len(briefs)} ({result.brief_id})"
            )
    return results


def score_result(result: RunResult, brief: dict, corpus: Corpus):
    """Score one `RunResult`. The report is the SUBMITTED one, always."""
    return score_run(brief, result.report, result.trace_jsonl, corpus)


def score_session(results, briefs, corpus: Corpus) -> list:
    by_id = {str(b.get("brief_id", "")): b for b in briefs}
    return [
        score_result(result, by_id.get(result.brief_id, {}), corpus)
        for result in results
    ]


# ---------------------------------------------------------------------------
# The preflight. Run it before ANY scored round.
# ---------------------------------------------------------------------------


def assert_provenance(results, *, where: str = "scored round") -> None:
    """Raise `PreflightError` if NO run recorded a decodable FINAL.

    This is the check that costs nothing and saves the round. The failure
    it catches — one dropped or unparseable `output_text` — converts an
    entire cohort's claims to `NOT_FROM_MODEL`: a ~55-point wipe, applied
    uniformly, with `gate_passed=True` and no other symptom anywhere. It
    is indistinguishable from "the briefs were hard" on the leaderboard,
    which is exactly why it has to fail here and loudly.
    """
    results = list(results)
    if not results:
        raise PreflightError(f"{where}: no runs to check")
    good = [r for r in results if r.provenance_ok]
    if good:
        return
    lines = [
        f"PREFLIGHT FAILED ({where}): not one of {len(results)} runs recorded a "
        f"model_call whose {MODEL_OUTPUT_FIELD} parses as a FINAL.",
        "Every claim in this batch would score NOT_FROM_MODEL. Do not run the "
        "scored round until this is fixed.",
    ]
    for result in results[:5]:
        lines.append(
            f"  - {result.brief_id or '(no id)'} seed={result.seed} "
            f"model_calls={result.model_calls} final_outputs={result.final_outputs} "
            f"stop={result.stop_reason or '-'} error={result.error or '-'}"
        )
    raise PreflightError("\n".join(lines))


def preflight(
    brief: dict,
    *,
    model,
    corpus: Corpus,
    middleware=None,
    seed: int = 42,
    config: RunnerConfig | None = None,
    clock=time.perf_counter,
) -> RunResult:
    """One live run, checked hard. Call this before every scored batch.

    Also verifies the trace gate, because a preflight that only checked
    provenance would happily green-light a batch that scores 0 for a
    different reason.
    """
    result = run_brief(
        brief,
        model=model,
        corpus=corpus,
        middleware=middleware,
        seed=seed,
        config=config,
        clock=clock,
    )
    ok, reason = result.gate()
    if not ok:
        raise PreflightError(
            f"PREFLIGHT FAILED: the trace does not conform ({reason}). "
            "Every entry scored against this runner would be zeroed by the gate."
        )
    assert_provenance([result], where=f"preflight on {result.brief_id or 'brief'}")
    return result


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------

#: Fields that are a wall clock by definition and therefore cannot be
#: byte-identical between two runs.
TIMING_FIELDS = ("elapsed_seconds",)


def strip_timing(jsonl: str) -> str:
    """The trace with its wall-clock fields removed.

    Two runs of the same `(brief, seed, model, middleware)` in two
    different processes are byte-identical after this. Used by the
    determinism tests and by `scripts/verify.py`.
    """
    out = []
    for line in jsonl.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            out.append(line)
            continue
        if isinstance(record, dict):
            for name in TIMING_FIELDS:
                record.pop(name, None)
            out.append(json.dumps(record, sort_keys=True))
        else:  # pragma: no cover - Trace never writes a non-object
            out.append(line)
    return "\n".join(out)


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DUMP_DISTINCT_DOCS_FLOOR",
    "DUMP_MODEL_CALL_CEILING",
    "EMPTY_OUTPUT_SENTINEL",
    "MAX_OUTPUT_TEXT_CHARS",
    "MAX_SEARCH_K",
    "MAX_TOKENS_PARAMS",
    "RUNNER_VERSION",
    "SCORED_PROMPT_ADDENDUM",
    "DumpSignature",
    "PreflightError",
    "ProvenanceModel",
    "ScoredBoundary",
    "RunAborted",
    "RunResult",
    "RunnerConfig",
    "RunnerError",
    "assert_provenance",
    "clamp_output_text",
    "clamp_search_k",
    "derive_seed",
    "dump_signature",
    "normalise_output",
    "shield_corpus",
    "preflight",
    "run_brief",
    "run_session",
    "score_result",
    "scored_boundary",
    "score_session",
    "strip_timing",
    "synthetic_abstain_report",
]
