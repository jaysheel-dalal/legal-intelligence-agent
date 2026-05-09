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
   removed by the Report Agent.

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

Run evaluation suite:
    python evaluation_v3.py

Run unit tests:
    pytest --tb=short -q

Run with mock (no API key needed):
    MockLLMClient activates
    automatically when
    ANTHROPIC_API_KEY is not set.

---

## Evaluation Results

Total: 19/20 (95%)

  Happy Path:  7/8  (88%)
  Exceptions:  4/4  (100%)
  Edge Cases:  4/4  (100%)
  Adversarial: 4/4  (100%)

Avg latency:  33.5 seconds
Total cost:   ~$1.21 per full run

---

## Test Results

109 / 109 unit tests passing

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
  jurisdiction + date context + 109 tests

---
