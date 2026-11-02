---
# Legal Intelligence Agent v1.0.0

A multi-agent system that reads any
business contract and answers plain
English questions about it —
grounded in the document, cited by
source, risk-scored, and
jurisdiction-aware.

Built for MIS6349 · UT Dallas
Spring 2026 · Jaysheel Dalal

---

## What it does

- Upload any business contract
  (PDF, DOCX, TXT)
- Ask in plain English:
  "Will I get paid if they fire me?"
- Four agents run in sequence to
  parse, analyze, detect gaps,
  and validate
- Every clause is risk-scored
  (CRITICAL / HIGH / MEDIUM / LOW),
  translated to plain English, and
  cited by position
- Missing required clauses are
  identified against a knowledge
  base template
- Contract jurisdiction and dates
  are extracted and passed as
  context to the LLM

---

## Scope

Supported contract types:
NDA, Employment Agreement,
SaaS/Vendor Agreement,
IP License Agreement,
Supply Agreement,
Independent Contractor Agreement,
Partnership Agreement

Jurisdiction: United States —
Federal, Delaware, California,
New York, Texas, Florida, Illinois

Not in scope (v1.0.0):
Court filings, securities
agreements, M&A agreements,
real estate leases, non-US
contracts, contracts over 75 pages

Not legal advice: The system
provides legal information only.
Always consult a qualified lawyer
before acting on any output.

---

## Architecture

Four agents run in sequence:

    Parser Agent
        ↓
    Analysis Agent
        ↓
    Gap Agent
        ↓
    Report Agent
        ↓
    Streamlit UI

Parser Agent — Loads PDF, detects
contract type, splits into chunks,
detects jurisdiction, extracts
effective date and expiry date
(including relative duration
calculation).

Analysis Agent — Sends chunks to
Claude Sonnet with query + intent +
jurisdiction + dates. Claude
assigns risk levels, confidence
scores, triggered flags, and plain
English explanations.

Gap Agent — Compares found clause
types against knowledge base
templates. Identifies missing
required clauses with severity.

Report Agent — Validates all
outputs. Removes results with
confidence below 0.6. Verifies
every clause has a source position.

---

## Model

Uses claude-sonnet-4-5

Chosen over Haiku (too weak,
returns identical MEDIUM for all
clauses) and Opus (overkill, 5x
more expensive). Sonnet correctly
follows complex risk-level
instructions and produces varied
confidence scores.

Average cost per query: $0.06-0.09

---

## Anti-Hallucination Measures

1. Parser-first clause typing —
   Claude never assigns clause_type.
   The parser uses deterministic
   keyword matching. Claude label
   is always replaced with the
   parser label.

2. Source position citations —
   Every clause must have a
   source_position (chunk number).
   No source = removed from output.

3. Confidence threshold — Results
   with confidence below 0.6 are
   removed, and so are results the
   model did not score at all. An
   absent confidence is absent, never
   a default: three layers used to
   substitute one (agent_core.analyze
   0.7, ClaudeLLMClient.validate 1.0,
   app.py 0.62-0.97 by risk level),
   so an unscored clause could reach
   the user shown as 97% confident
   and the filter could not reject
   it. The gate now lives in one
   place,
   legal_intelligence_agent/
   confidence.py. The UI still shows
   a derived number when the model
   gave none, but under a separate
   key and labelled "est.", so it
   cannot be read back as a model
   score.

4. Grounding rule in system prompt
   — Never invent a clause. Only
   return what exists in the
   document.

5. Gap detection from JSON templates
   — Missing clauses identified by
   comparing parsed clause types
   against knowledge base JSON,
   not by asking Claude to guess.

---

## How to run

Install dependencies:
    pip install -r requirements.txt

Set API key (Windows PowerShell):
    $env:ANTHROPIC_API_KEY = "your-key"

Run the web interface:
    streamlit run app.py

Run evaluation suites:
    python evaluation_v3.py
    python evaluation_factual.py

Run unit tests:
    pytest --tb=short -q

Run with mock (no API key needed):
    MockLLMClient activates
    automatically when
    ANTHROPIC_API_KEY is not set.

---

## Evaluation Results (structural)

evaluation_v3.py runs 30 scenarios
across five zones: Happy Path (8),
Exceptions (4), Edge Cases (4),
Adversarial (4), Premortem Risk (10).

What it checks is structural, not
factual. Per scenario:

- the detected contract type matches
- total_clauses_found meets a minimum
  (which is 0 for 16 of the 30, making
  the check a no-op for those)
- the report text is non-empty
- no phrase from an unsafe-response
  blocklist appears

A scenario therefore passes on output
that is confidently wrong, so long as
it is well-formed. Each scenario also
defines expected_intent, which
score_scenario never reads. For
correctness measures, see Factual
Correctness Evaluation below.

What the committed logs actually show:

    logs/eval_v3_20260502_133442.json
    logs/eval_v3_20260502_135207.json

    Live Claude · 20 scenarios · 19/20
    (95%)

    Happy Path:   7/8
    Exceptions:   4/4
    Edge Cases:   4/4
    Adversarial:  4/4

    Failure: V3-HP-03 (IP ownership
    terms on an IP licensing agreement)

Both logs predate the 10 Premortem Risk
scenarios, which have no committed run
of their own. No 30-scenario live run is
logged in this repo, so the 30/30 figure
previously quoted here was not
reproducible from the committed
evidence.

---

## Factual Correctness Evaluation

    python evaluation_factual.py            (mock, no API key)
    python evaluation_factual.py --strict   (exit 1 on regression)
    python evaluation_factual.py --live     (real Claude, costs money)

The suite above counts scenarios that
produced plausible-looking output. It
checks contract type, a minimum clause
count, non-empty text, and a blocklist
of unsafe phrases — all of which pass
for output that is confidently wrong.
Its per-scenario expected_intent is
never read.

evaluation_factual.py scores whether
the output is true, against gold labels
in evaluation_gold_labels.json written
by hand for the committed fixtures
(tests/fixtures/*.txt, not contracts/*,
which is gitignored and so cannot be
reproduced or run in CI).

Three measures:

1. Clause typing — precision / recall /
   F1 over (source_position,
   clause_type) pairs.
2. Source grounding — every cited
   source_position must exist, the
   quoted clause text must actually
   appear at it, and that passage must
   match the gold anchor phrase.
3. Intent — expected_intent compared
   against the pipeline's intent.

A fourth check validates the gold
labels themselves: every anchor phrase
must still appear in the chunk it
describes, so the ground truth fails
loudly if the fixtures change.

Measured (mock client, deterministic):

                     before    after
    Clause typing  P  0.692    0.923
                   R  0.692    0.923
                   F1 0.692    0.923
                      tp 9     tp 12
                      fp 4     fp 1
                      fn 4     fn 1
    Grounding         13/13    13/13
    Intent            6/7      7/7

    before: logs/eval_factual_
            20261102_144318.json
    after:  logs/eval_factual_
            20261102_145716.json

Unlike the 30/30 suite, this one
reported real defects, and the "after"
column is what fixing them looks like.

Three of the five were first-match
ordering bugs: a specific clause type
losing to a broader one that the text
mentions only in passing. A
limitation-of-liability clause was
reported as "payment" because it says
"fees paid or payable"; an EXCLUSIONS
carve-out was reported as
"confidentiality"; a RETURN OF
MATERIALS clause was reported as
"term_and_termination". The elif chain
in _detect_clause_type is now an
explicit ordered rule table
(CLAUSE_TYPE_RULES) running
specific-to-general, with the
orderings that matter commented. The
dead "termination" branch, unreachable
because "term_and_termination" matched
"terminat" first, was removed.

The intent miss routed every question
mentioning a vendor to "termination",
because the signal "end" matched inside
"vendor". Signal matching is now
boundary-anchored
(legal_intelligence_agent/
text_signals.py): "end" still matches
end / ends / ending, but not vendor,
and "own" no longer matches inside
"unknown".

One mislabel remains, deliberately: the
NDA title block is still reported as
"confidentiality" because it contains
the words "Non-Disclosure Agreement".
No rule ordering fixes that — it needs
a notion of "this chunk states no
obligation" — so it stays visible in
the score rather than being rounded
away.

Baselines are recorded in
evaluation_factual.BASELINE so --strict
fails on regression rather than on
these known defects.
tests/test_evaluation_factual.py proves
each scorer rejects the failure it
claims to catch, including a fabricated
"Section 99" citation.

---

## Test Results

187 / 187 unit tests passing
(109 pipeline + 24 factual
correctness eval + 54 confidence
gate)

---

## Knowledge Base

Contract templates (gap detection):
  nda.json
  employment.json
  saas.json
  vendor.json
  ip_license.json
  contractor.json
  partnership.json

Legal frameworks:
  employment.json
  nda_law.json
  ip_law.json
  contract_law.json
  ccpa.json
  payment_liability.json

---

## Human in the Loop

The system flags queries with 1 or
more CRITICAL clauses as requiring
immediate attention. This count is
shown at the top of every result.
In production this triggers an
escalation workflow to a legal
reviewer. In v1.0.0 it is a soft
signal.

---

## Project Status

Project 1 — Done: 6-layer agent
Project 2 — Done: evaluation
  framework + approval gate
Project 3 — Done: 4-agent
  sequential pipeline +
  jurisdiction + date context + 187 tests

---
