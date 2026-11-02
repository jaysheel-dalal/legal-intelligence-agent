"""Factual-correctness evaluation for the legal intelligence pipeline.

evaluation_v3.py scores whether the pipeline produced *something*: the right
contract type, at least N clauses, a non-empty report, no blocklisted phrase.
Those checks pass for output that is confidently wrong, and its per-scenario
expected_intent is never read at all.

This suite scores whether the pipeline produced something *true*, against
hand-written gold labels in evaluation_gold_labels.json:

  1. Clause typing (precision / recall / F1)
     Set-retrieval over (source_position, clause_type) pairs. Precision and
     recall are not the same number here: the pipeline can report a chunk the
     gold labels do not cover (precision-only penalty) and can drop a gold
     chunk entirely via the 0.6 confidence filter (recall-only penalty).

  2. Source grounding (citation integrity)
     Every clause the pipeline reports carries a source_position. For each one
     this re-parses the fixture independently and asks: does that position
     exist, does the quoted clause text actually appear at it, and does the
     passage there match the gold label's anchor phrase? A clause that cites a
     position it cannot support is the hallucination failure mode the
     blocklist in evaluation_v3.py cannot see.

  3. Intent (accuracy)
     expected_intent is actually compared against the pipeline's intent, which
     evaluation_v3.py defines for all 30 scenarios and never checks.

  4. Gold-label integrity
     Each gold anchor phrase must appear in the chunk it claims to describe, so
     the ground truth fails loudly if the fixtures are edited underneath it.

Labels are anchored to tests/fixtures/*.txt, not contracts/*.pdf: contracts/ is
gitignored, so a PDF-based eval cannot be reproduced by anyone else or run in
CI. Runs on MockLLMClient by default -- deterministic, no API key, no spend.

Usage:
    python evaluation_factual.py                 # mock client (CI-safe)
    python evaluation_factual.py --live          # real Claude, costs money
    python evaluation_factual.py --strict        # exit 1 on any regression
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

GOLD_PATH = "evaluation_gold_labels.json"

# Current measured scores, so --strict fails on a regression rather than on the
# defects the gold labels already document. Raise these as the pipeline improves.
BASELINE = {
    "clause_type_f1": 0.69,
    "grounding_rate": 1.00,
    "intent_accuracy": 0.85,
}


def _norm(text):
    """Collapse whitespace and case so substring checks survive reformatting."""
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def parse_fixture(fixture_path):
    """Parse the fixture directly, as the independent reference for grounding.

    Deliberately not the pipeline's own chunk list: a citation has to be checked
    against the source document, not against the pipeline's echo of it.
    """
    from legal_intelligence_agent.document_parser import DocumentParser

    doc = DocumentParser().parse_document(document_path=fixture_path)
    return doc, {c["position"]: c for c in doc["chunks"]}


def check_gold_integrity(gold_clauses, chunk_by_pos):
    """Fail loudly if the gold labels no longer describe the fixture."""
    problems = []
    for gold in gold_clauses:
        pos = gold["source_position"]
        chunk = chunk_by_pos.get(pos)
        if chunk is None:
            problems.append(
                f"gold position {pos} does not exist in the parsed fixture "
                f"(document has {len(chunk_by_pos)} chunks)"
            )
            continue
        anchor = gold.get("must_contain")
        if anchor and _norm(anchor) not in _norm(chunk["text"]):
            problems.append(
                f"gold position {pos} anchor {anchor!r} not found in that chunk"
            )
    return problems


def run_pipeline(llm_factory, query, fixture_path, log_dir):
    from agents.orchestrator import Orchestrator

    orch = Orchestrator(
        llm_client=llm_factory(),
        log_dir=log_dir,
        output_format="dict",
    )
    return orch.run(query=query, file_path=fixture_path)


def reported_clauses(result):
    """The clause list the pipeline stands behind, after its own confidence filter."""
    agent_result = result.get("agent_result") or {}
    return agent_result.get("analyzed_clauses") or []


def score_clause_types(gold_clauses, reported):
    """Set-retrieval scoring over (source_position, clause_type) pairs."""
    gold_by_pos = {g["source_position"]: g for g in gold_clauses}

    tp = fp = fn = 0
    errors = []
    matched_positions = set()

    for clause in reported:
        pos = clause.get("source_position")
        got = clause.get("clause_type", "")
        gold = gold_by_pos.get(pos)

        if gold is None:
            fp += 1
            errors.append(
                {
                    "kind": "unlabelled_position",
                    "source_position": pos,
                    "reported": got,
                    "detail": "pipeline reported a chunk the gold labels do not cover",
                }
            )
            continue

        allowed = {gold["gold_clause_type"]} | set(gold.get("acceptable", []))
        if got in allowed:
            tp += 1
            matched_positions.add(pos)
        else:
            # One prediction is both a wrong label (FP) and a gold item left
            # unfound (FN), which is why precision and recall both drop.
            fp += 1
            fn += 1
            matched_positions.add(pos)
            errors.append(
                {
                    "kind": "wrong_clause_type",
                    "source_position": pos,
                    "reported": got,
                    "expected": gold["gold_clause_type"],
                    "detail": gold.get("rationale", ""),
                }
            )

    for pos, gold in sorted(gold_by_pos.items()):
        if pos not in matched_positions:
            fn += 1
            errors.append(
                {
                    "kind": "clause_not_reported",
                    "source_position": pos,
                    "expected": gold["gold_clause_type"],
                    "detail": "gold clause absent from output (dropped before reporting)",
                }
            )

    scores = prf(tp, fp, fn)
    scores["errors"] = errors
    return scores


def score_grounding(gold_clauses, reported, chunk_by_pos):
    """Verify every citation resolves to text that supports the clause."""
    gold_by_pos = {g["source_position"]: g for g in gold_clauses}

    checked = 0
    grounded = 0
    failures = []

    for clause in reported:
        checked += 1
        pos = clause.get("source_position")

        if pos is None or not isinstance(pos, int):
            failures.append(
                {
                    "kind": "missing_citation",
                    "source_position": pos,
                    "detail": "clause reported with no usable source_position",
                }
            )
            continue

        chunk = chunk_by_pos.get(pos)
        if chunk is None:
            failures.append(
                {
                    "kind": "dangling_citation",
                    "source_position": pos,
                    "detail": (
                        f"cites position {pos} but the document has only "
                        f"{len(chunk_by_pos)} chunks"
                    ),
                }
            )
            continue

        source_text = _norm(chunk["text"])
        quoted = _norm(clause.get("text", ""))

        if not quoted:
            failures.append(
                {
                    "kind": "empty_quote",
                    "source_position": pos,
                    "detail": "clause cites a position but quotes no text from it",
                }
            )
            continue

        if quoted not in source_text:
            failures.append(
                {
                    "kind": "text_mismatch",
                    "source_position": pos,
                    "detail": (
                        "quoted clause text does not appear at the cited position: "
                        f"{quoted[:90]!r}"
                    ),
                }
            )
            continue

        gold = gold_by_pos.get(pos)
        anchor = gold.get("must_contain") if gold else None
        if anchor and _norm(anchor) not in source_text:
            failures.append(
                {
                    "kind": "wrong_passage",
                    "source_position": pos,
                    "detail": (
                        f"cited position does not contain the gold anchor {anchor!r}"
                    ),
                }
            )
            continue

        grounded += 1

    rate = (grounded / checked) if checked else 0.0
    return {
        "checked": checked,
        "grounded": grounded,
        "rate": round(rate, 3),
        "failures": failures,
    }


def score_intent(expected, result):
    got = result.get("intent", "")
    return {
        "expected": expected,
        "got": got,
        "correct": got == expected,
    }


def evaluate(use_mock=True, gold_path=GOLD_PATH, log_dir="logs/factual",
             report_dir="logs"):
    if use_mock:
        from main_v3 import MockLLMClient

        llm_factory = MockLLMClient
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set (required for --live).")
            return None
        from main_v3 import ClaudeLLMClient

        llm_factory = lambda: ClaudeLLMClient(api_key=api_key)

    with open(gold_path, encoding="utf-8") as f:
        gold_data = json.load(f)

    print()
    print("=" * 68)
    print("FACTUAL CORRECTNESS EVALUATION")
    print("=" * 68)
    print(f"Gold labels: {gold_path}")
    print(f"Mode: {'MOCK' if use_mock else 'LIVE CLAUDE'}")
    print("=" * 68)

    fixture_reports = []
    tot_tp = tot_fp = tot_fn = 0
    tot_checked = tot_grounded = 0
    tot_intent = tot_intent_ok = 0
    integrity_problems = []

    for spec in gold_data["fixtures"]:
        fixture = spec["fixture"]
        gold_clauses = spec["clauses"]

        print()
        print("-" * 68)
        print(f"FIXTURE: {fixture}")
        print("-" * 68)

        doc, chunk_by_pos = parse_fixture(fixture)

        # 4. Gold-label integrity, before trusting any score derived from it.
        problems = check_gold_integrity(gold_clauses, chunk_by_pos)
        if problems:
            integrity_problems.extend(f"{fixture}: {p}" for p in problems)
            print("  GOLD LABEL INTEGRITY: FAILED")
            for p in problems:
                print(f"    ! {p}")
        else:
            print(f"  Gold label integrity: {len(gold_clauses)} anchors verified in source")

        got_type = doc.get("contract_type", "")
        want_type = spec.get("gold_contract_type", "")
        print(
            f"  Contract type: {got_type} "
            f"({'ok' if got_type == want_type else 'expected ' + want_type})"
        )

        # 1 + 2. Clause typing and grounding, on the coverage query.
        coverage_result = run_pipeline(
            llm_factory, spec["coverage_query"], fixture, log_dir
        )
        reported = reported_clauses(coverage_result)

        ct = score_clause_types(gold_clauses, reported)
        gr = score_grounding(gold_clauses, reported, chunk_by_pos)

        tot_tp += ct["tp"]
        tot_fp += ct["fp"]
        tot_fn += ct["fn"]
        tot_checked += gr["checked"]
        tot_grounded += gr["grounded"]

        print()
        print(
            f"  Clause typing:  tp={ct['tp']} fp={ct['fp']} fn={ct['fn']}  "
            f"precision={ct['precision']}  recall={ct['recall']}  f1={ct['f1']}"
        )
        for err in ct["errors"]:
            pos = err["source_position"]
            if err["kind"] == "wrong_clause_type":
                print(
                    f"    x pos={pos}  reported '{err['reported']}'  "
                    f"expected '{err['expected']}'"
                )
            elif err["kind"] == "clause_not_reported":
                print(f"    x pos={pos}  '{err['expected']}' never reported")
            else:
                print(f"    x pos={pos}  {err['detail']}")

        print(
            f"  Grounding:      {gr['grounded']}/{gr['checked']} citations "
            f"resolve to supporting text  (rate={gr['rate']})"
        )
        for fail in gr["failures"]:
            print(f"    x pos={fail['source_position']}  {fail['kind']}: {fail['detail']}")

        # 3. Intent, per query.
        print()
        intent_results = []
        for q in spec["queries"]:
            result = run_pipeline(llm_factory, q["query"], fixture, log_dir)
            scored = score_intent(q["expected_intent"], result)
            scored["query"] = q["query"]
            intent_results.append(scored)
            tot_intent += 1
            if scored["correct"]:
                tot_intent_ok += 1
            flag = "ok  " if scored["correct"] else "MISS"
            print(
                f"  Intent {flag}  expected={scored['expected']:<16} "
                f"got={str(scored['got']):<16} {q['query']}"
            )

        fixture_reports.append(
            {
                "fixture": fixture,
                "contract_type": {"expected": want_type, "got": got_type},
                "gold_integrity_problems": problems,
                "clause_typing": ct,
                "grounding": gr,
                "intent": intent_results,
            }
        )

    overall_ct = prf(tot_tp, tot_fp, tot_fn)
    grounding_rate = (tot_grounded / tot_checked) if tot_checked else 0.0
    intent_accuracy = (tot_intent_ok / tot_intent) if tot_intent else 0.0

    print()
    print("=" * 68)
    print("OVERALL")
    print("=" * 68)
    print(
        f"  Clause typing   precision={overall_ct['precision']}  "
        f"recall={overall_ct['recall']}  f1={overall_ct['f1']}  "
        f"(tp={tot_tp} fp={tot_fp} fn={tot_fn})"
    )
    print(
        f"  Grounding       {tot_grounded}/{tot_checked} "
        f"({grounding_rate * 100:.0f}%)"
    )
    print(
        f"  Intent          {tot_intent_ok}/{tot_intent} "
        f"({intent_accuracy * 100:.0f}%)"
    )
    if integrity_problems:
        print(f"  GOLD INTEGRITY  {len(integrity_problems)} problem(s) -- scores unreliable")
    print("=" * 68)

    report = {
        "timestamp": datetime.now().isoformat(),
        "suite": "factual_correctness",
        "mode": "mock" if use_mock else "live",
        "gold_labels": gold_path,
        "overall": {
            "clause_typing": overall_ct,
            "grounding": {
                "checked": tot_checked,
                "grounded": tot_grounded,
                "rate": round(grounding_rate, 3),
            },
            "intent": {
                "total": tot_intent,
                "correct": tot_intent_ok,
                "accuracy": round(intent_accuracy, 3),
            },
        },
        "gold_integrity_problems": integrity_problems,
        "fixtures": fixture_reports,
    }

    os.makedirs(report_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"eval_factual_{stamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")

    return report


def check_regressions(report):
    """Compare against BASELINE. Returns a list of regression strings."""
    overall = report["overall"]
    measured = {
        "clause_type_f1": overall["clause_typing"]["f1"],
        "grounding_rate": overall["grounding"]["rate"],
        "intent_accuracy": overall["intent"]["accuracy"],
    }
    regressions = []
    for key, floor in BASELINE.items():
        if measured[key] < floor:
            regressions.append(f"{key}: {measured[key]} < baseline {floor}")
    if report["gold_integrity_problems"]:
        regressions.append(
            f"gold label integrity: {len(report['gold_integrity_problems'])} problem(s)"
        )
    return regressions


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--live",
        action="store_true",
        help="use the real Claude client instead of MockLLMClient (costs money)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if any metric falls below its recorded baseline",
    )
    parser.add_argument("--gold", default=GOLD_PATH, help="path to gold labels JSON")
    args = parser.parse_args()

    report = evaluate(use_mock=not args.live, gold_path=args.gold)
    if report is None:
        return 2

    if args.strict:
        regressions = check_regressions(report)
        if regressions:
            print()
            print("REGRESSIONS vs baseline:")
            for r in regressions:
                print(f"  - {r}")
            return 1
        print()
        print("No regressions against baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
