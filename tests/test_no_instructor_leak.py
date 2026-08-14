"""The distribution guard: prove this bundle carries no scored-round material.

This repository is PUBLIC and 100+ people clone it. The scored round runs
on briefs nobody outside the instructor's machine has seen, and a single
leaked question, claim or verdict invalidates that round for everyone at
once — retroactively, with no way to recall the copies. So the boundary
gets a test rather than a convention, and the test runs in the student
bundle where the mistake would actually land.

Four things are checked, in increasing order of subtlety.

1. NO INSTRUCTOR TREE. No `instructor/` directory, no `briefs_private.json`,
   no `reference/` package: the five reference layer implementations, the
   oracle, the adversary harness and the scored briefs all live there.
2. NO IMPORT OF ONE. A test or script that says `from instructor.reference
   import ...` is either shipping the answer key or crashing on every
   student machine. Both are release blockers, and the second is how the
   first usually gets noticed too late.
3. NO SCORED BRIEF IDS. The scored set numbers its briefs `prv-NN-...`;
   that pattern appears nowhere in a clean bundle.
4. NO SCORED BRIEF TEXT — the one that actually needs a technique.

THE TECHNIQUE, AND WHY IT IS NOT PLAINTEXT
==========================================

A leak test that shipped the secrets it looks for would BE the leak. So
nothing here stores any scored brief's text. At build time, the packaging
step read the private brief file and reduced every question, claim,
key term, verdict phrase, verdict id, brief id, refined query and
instructor note to the set of its 6-word n-grams ("shingles"), normalised
(NFC, casefolded, word characters only), then stored a salted, truncated
SHA-256 of each in `tests/private_fingerprints.json`. Recovering a
sentence from those is a preimage attack per n-gram; matching one is a
set intersection. The test recomputes shingles for every text file in the
tree and intersects.

Shingles rather than whole-string hashes, because a leak is rarely a
whole file: someone pastes ONE question into a docstring while debugging.
An n-gram catches the fragment; a whole-string hash would not.

WHY THERE ARE TWO TIERS
=======================

The scored briefs quote the corpus, and the corpus ships. A scored
brief's `claim` IS a verbatim line of some `data/corpus/doc-XXXX.json`,
by construction — that is not a leak, that is the lab working. What
distinguishes a leak from the corpus doing its job is WHERE the sentence
appears.

* `tier_all` — n-grams that occur in the private set and NOWHERE in this
  bundle: questions, verdict phrases, brief ids, refined queries,
  instructor notes. Any occurrence, in any file, is a leak. The set is
  computed as a DIFFERENCE against the shipped bundle, so shared
  boilerplate ("Trong kỳ báo cáo, ...") is subtracted rather than
  producing a false alarm — what is left is what is unique to the scored
  set.
* `tier_prose` — n-grams of the scored CLAIMS and key terms that are not
  already elsewhere in the bundle. These are checked everywhere EXCEPT
  `data/corpus/`, whose documents are the sentences' legitimate home.

Both tiers are empty-by-construction on the bundle as built: if either
were non-empty at build time the packaging step would have failed. So a
failure here always means something changed after packaging.

WHAT THIS TEST CANNOT DO
========================
It cannot catch a paraphrase, and it cannot catch a leak expressed as
code (a script that reconstructs a scored brief from the corpus). It is a
tripwire on the most likely accident — copy-paste — not a proof of
secrecy. The proof is that `instructor/` was never copied in the first
place, which is what tests 1 and 2 pin.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINTS_PATH = Path(__file__).resolve().parent / "private_fingerprints.json"

#: Directories that are not part of the distribution.
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    "runs",
    "node_modules",
}

#: Suffixes worth scanning. Everything human-authored in this lab is text.
TEXT_SUFFIXES = {
    "",
    ".py",
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".cfg",
    ".ini",
    ".toml",
    ".yaml",
    ".yml",
    ".html",
    ".rst",
    ".sh",
}

#: This file names the forbidden things in order to look for them, so it
#: is the one file the text scans skip. Nothing else may be exempted.
SELF = Path(__file__).name

_WORD_RE = re.compile(r"\w+", re.UNICODE)

#: `from instructor...` / `import instructor...`, however it is spelled.
_IMPORT_RE = re.compile(
    r"^[ \t]*(?:from[ \t]+instructor\b|import[ \t]+instructor\b)", re.MULTILINE
)

#: The scored set's brief-id scheme. Publishing the SCHEME costs nothing —
#: `arena/briefs.py` documents it — but an actual id would mean a brief
#: came with it.
_PRIVATE_ID_RE = re.compile(r"\bprv-\d{2}-[a-z0-9-]+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Shingling — identical to the build-time derivation, or the sets do not
# line up. Keep the two in step if either ever changes.
# ---------------------------------------------------------------------------


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(unicodedata.normalize("NFC", str(text)).casefold())


def _shingles(text: str, size: int) -> set[str]:
    words = _words(text)
    if not words:
        return set()
    if len(words) <= size:
        return {" ".join(words)}
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def _fingerprint(shingle: str, salt: str) -> str:
    import hashlib

    return hashlib.sha256((salt + "\x1f" + shingle).encode("utf-8")).hexdigest()[:16]


def _text_files():
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        yield path


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


@pytest.fixture(scope="module")
def fingerprints() -> dict:
    assert FINGERPRINTS_PATH.exists(), (
        "tests/private_fingerprints.json is missing. It is the only thing that "
        "makes the leak scan below mean anything — restore it rather than "
        "deleting this test."
    )
    data = json.loads(FINGERPRINTS_PATH.read_text(encoding="utf-8"))
    assert data["tier_all"] and data["tier_prose"], "empty fingerprint set"
    assert isinstance(data["shingle"], int) and data["shingle"] >= 4
    return data


# ---------------------------------------------------------------------------
# 1. No instructor tree
# ---------------------------------------------------------------------------


def test_no_instructor_package_ships():
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in REPO_ROOT.rglob("*")
        if p.name in {"instructor", "briefs_private.json"}
        or (p.is_dir() and p.name == "reference" and (p / "oracle.py").exists())
    ]
    assert offenders == [], offenders


def test_the_scored_brief_file_is_absent_and_the_code_says_so_calmly():
    """`arena/briefs.py` documents `FileNotFoundError` as the NORMAL state
    on a student machine. Anything else — a stub file, an empty list —
    would let a scored round start against zero briefs."""
    from arena.briefs import PRIVATE_BRIEFS_PATH, load_private_briefs

    assert not PRIVATE_BRIEFS_PATH.exists(), str(PRIVATE_BRIEFS_PATH)
    with pytest.raises(FileNotFoundError):
        load_private_briefs()


def test_no_module_imports_the_instructor_package():
    offenders = []
    for path in _text_files():
        if path.name == SELF or path.suffix != ".py":
            continue
        text = _read(path)
        if text and _IMPORT_RE.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], offenders


#: Paths into the instructor pack, in the two spellings that matter.
_PACK_NEEDLES = ("instructor/reference", "instructor.reference", "briefs_private")


def test_nothing_outside_the_frozen_scaffold_names_the_instructor_pack():
    """A string like `"instructor/reference/critic.py"` in a script or a
    test is a dependency this bundle cannot satisfy — it either crashes on
    every student machine or, worse, tells a reader exactly what to go
    looking for.

    `arena/` is exempt and the next test explains why: it is frozen
    scaffold, it is identical byte-for-byte in the instructor tree, and it
    names the boundary DELIBERATELY — `PRIVATE_BRIEFS_PATH` has to point
    somewhere for `load_private_briefs()` to raise the documented
    `FileNotFoundError`.
    """
    offenders = []
    for path in _text_files():
        relative = path.relative_to(REPO_ROOT)
        if path.name == SELF or relative.parts[0] == "arena":
            continue
        text = _read(path)
        if text is None:
            continue
        hits = [n for n in _PACK_NEEDLES if n in text]
        if hits:
            offenders.append((str(relative), hits))
    assert offenders == [], offenders


def test_the_frozen_scaffolds_own_mentions_are_pinned():
    """The exemption above is bounded: `arena/` may mention the pack only
    in the places it already does. A NEW mention means someone edited
    frozen code, which `scripts/verify.py` would also catch by MD5 — this
    just says which file to look in.
    """
    counts = {}
    for path in sorted((REPO_ROOT / "arena").glob("*.py")):
        text = _read(path) or ""
        hits = sum(text.count(n) for n in _PACK_NEEDLES)
        if hits:
            counts[path.name] = hits
    assert counts == {"briefs.py": 2, "runner.py": 1}, counts


# ---------------------------------------------------------------------------
# 2. No scored brief ids
# ---------------------------------------------------------------------------


def test_no_scored_brief_id_appears_anywhere():
    offenders = []
    for path in _text_files():
        if path.name == SELF:
            continue
        text = _read(path)
        if text is None:
            continue
        found = sorted({m.group(0) for m in _PRIVATE_ID_RE.finditer(text)})
        if found:
            offenders.append((str(path.relative_to(REPO_ROOT)), found[:3]))
    assert offenders == [], offenders


# ---------------------------------------------------------------------------
# 3. No scored brief TEXT — the fingerprint scan
# ---------------------------------------------------------------------------


def scan(wanted, salt: str, size: int, *, skip_corpus: bool = False) -> list:
    """Every text file whose shingles intersect `wanted`, as (path, hits).

    The one place file walking, normalisation, hashing and intersection
    happen — so the self-test at the bottom exercises exactly the code the
    real assertions run.
    """
    offenders = []
    for path in _text_files():
        relative = path.relative_to(REPO_ROOT)
        if path.name == SELF:
            continue
        if skip_corpus and relative.parts[:2] == ("data", "corpus"):
            continue
        text = _read(path)
        if text is None:
            continue
        hits = {_fingerprint(s, salt) for s in _shingles(text, size)} & set(wanted)
        if hits:
            offenders.append((relative.as_posix(), len(hits)))
    return offenders


def test_no_scored_brief_text_appears_anywhere(fingerprints):
    """Questions, verdict phrases, refined queries and instructor notes.

    Any hit is a leak: every n-gram in this tier was verified absent from
    the whole bundle when the bundle was built.
    """
    offenders = scan(
        fingerprints["tier_all"], fingerprints["salt"], fingerprints["shingle"]
    )
    assert offenders == [], (
        "SCORED-ROUND TEXT FOUND IN THE STUDENT BUNDLE. Do not commit. "
        f"Files and how many n-grams matched: {offenders}"
    )


def test_no_scored_claim_text_appears_outside_the_corpus(fingerprints):
    """The scored claims are verbatim corpus lines, so `data/corpus/` is
    exempt — that is where those sentences live and the corpus is public
    by design. A scored claim quoted in a test, a script, a README or a
    brief file is a different thing entirely: it is the pairing of the
    sentence with the fact that it MATTERS, which is the whole secret.
    """
    offenders = scan(
        fingerprints["tier_prose"],
        fingerprints["salt"],
        fingerprints["shingle"],
        skip_corpus=True,
    )
    assert offenders == [], (
        "SCORED CLAIM TEXT FOUND OUTSIDE data/corpus/. Do not commit. "
        f"Files and how many n-grams matched: {offenders}"
    )


# ---------------------------------------------------------------------------
# 4. The guard guards itself
# ---------------------------------------------------------------------------


def test_the_fingerprint_file_carries_no_plaintext(fingerprints):
    """A fingerprint is 16 hex characters and nothing else. If a future
    edit ever stores the strings themselves "for readability", this fails
    before the file reaches anyone."""
    hex16 = re.compile(r"^[0-9a-f]{16}$")
    for tier in ("tier_all", "tier_prose"):
        for value in fingerprints[tier]:
            assert hex16.match(value), (tier, value)
    raw = FINGERPRINTS_PATH.read_text(encoding="utf-8")
    assert not _PRIVATE_ID_RE.search(raw)
    # No Vietnamese diacritics: the scored set is Vietnamese prose, so any
    # would mean plaintext crept in.
    assert not re.search(r"[à-ỹÀ-Ỹ]", raw)


def test_the_scan_actually_covers_the_bundle():
    """A scan that silently walked zero files would pass every assertion
    above. Pin the floor so a bad `SKIP_DIRS` edit cannot disarm it."""
    scanned = [p.relative_to(REPO_ROOT).as_posix() for p in _text_files()]
    assert len(scanned) > 100, len(scanned)
    for expected in ("scripts/run_practice.py", "data/briefs_public.json",
                     "harness/agent.py", "arena/scorer.py"):
        assert expected in scanned, expected


def test_the_scan_would_actually_notice(fingerprints):
    """Arm the tripwire and step on it.

    A leak test nobody has ever seen fail is indistinguishable from a leak
    test that cannot fail. This plants a file inside the tree containing a
    KNOWN, harmless sentence, fingerprints that sentence the same way the
    build step fingerprints a scored brief, and asserts the scan finds the
    file — then removes it. No scored text is involved: the mechanism is
    what is under test.
    """
    salt, size = fingerprints["salt"], fingerprints["shingle"]
    canary = (
        "khong phai brief that day chi la mot cau moi de kiem tra rang co che "
        "quet dau van hoat dong binh thuong nhu mong doi"
    )
    wanted = {_fingerprint(s, salt) for s in _shingles(canary, size)}
    assert not (wanted & set(fingerprints["tier_all"])), "canary must be foreign"

    planted = REPO_ROOT / "tests" / "_leak_canary_tmp.txt"
    assert not planted.exists()
    try:
        planted.write_text(canary, encoding="utf-8")
        offenders = scan(wanted, salt, size)
        assert [name for name, _ in offenders] == ["tests/_leak_canary_tmp.txt"], offenders
    finally:
        planted.unlink(missing_ok=True)

    # ...and it goes quiet again once the file is gone.
    assert scan(wanted, salt, size) == []
