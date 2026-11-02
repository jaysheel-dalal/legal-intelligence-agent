"""Tests for the factual-correctness eval in evaluation_factual.py.

An eval nobody has shown can fail is not evidence of anything, so these tests
do two jobs:

  - prove each scorer rejects the failure it claims to catch (fabricated
    citations, misquoted text, wrong labels, dropped clauses)
  - lock the measured scores on the committed fixtures, so a pipeline change
    that silently degrades clause typing, grounding, or intent breaks CI
"""

import inspect
import json
import re

import pytest

import evaluation_factual as ef
from legal_intelligence_agent.document_parser import DocumentParser


@pytest.fixture(scope="module")
def gold():
    with open(ef.GOLD_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def nda_chunks():
    _, chunk_by_pos = ef.parse_fixture("tests/fixtures/clean_nda.txt")
    return chunk_by_pos


def _emittable_clause_types():
    """Clause-type labels DocumentParser._detect_clause_type can actually return."""
    src = inspect.getsource(DocumentParser._detect_clause_type)
    return set(re.findall(r'return "([A-Za-z_]+)"', src))


class TestGoldLabelIntegrity:
    def test_every_anchor_appears_in_the_chunk_it_describes(self, gold):
        """Gold labels must still describe the fixtures they were written against."""
        for spec in gold["fixtures"]:
            _, chunk_by_pos = ef.parse_fixture(spec["fixture"])
            problems = ef.check_gold_integrity(spec["clauses"], chunk_by_pos)
            assert problems == [], f"{spec['fixture']}: {problems}"

    def test_every_gold_label_is_reachable(self, gold):
        """A gold label the classifier can never emit would be an unfalsifiable test.

        evaluation_v3.py scenario V3-EC-03 expects intent 'governing_law', which
        is not in the intent vocabulary at all -- invisible only because that
        file never checks expected_intent. This guards against the same mistake.
        """
        emittable = _emittable_clause_types()
        for spec in gold["fixtures"]:
            for clause in spec["clauses"]:
                labels = {clause["gold_clause_type"]} | set(clause.get("acceptable", []))
                unreachable = labels - emittable
                assert not unreachable, (
                    f"{spec['fixture']} pos={clause['source_position']}: "
                    f"{unreachable} cannot be produced by _detect_clause_type"
                )

    def test_integrity_check_catches_a_stale_anchor(self, nda_chunks):
        problems = ef.check_gold_integrity(
            [{"source_position": 1, "must_contain": "a phrase that is not in the NDA"}],
            nda_chunks,
        )
        assert len(problems) == 1
        assert "not found" in problems[0]

    def test_integrity_check_catches_a_position_off_the_end(self, nda_chunks):
        problems = ef.check_gold_integrity(
            [{"source_position": 99, "must_contain": "anything"}], nda_chunks
        )
        assert len(problems) == 1
        assert "does not exist" in problems[0]


class TestGrounding:
    def test_real_pipeline_output_is_grounded(self, gold, nda_chunks, tmp_path):
        """The honest case: every citation resolves to supporting text."""
        spec = gold["fixtures"][0]
        from main_v3 import MockLLMClient

        # tmp_path, not logs/: a test must not write into the tracked tree.
        result = ef.run_pipeline(
            MockLLMClient, spec["coverage_query"], spec["fixture"], str(tmp_path)
        )
        scored = ef.score_grounding(
            spec["clauses"], ef.reported_clauses(result), nda_chunks
        )
        assert scored["checked"] > 0
        assert scored["failures"] == []
        assert scored["rate"] == 1.0

    def test_rejects_fabricated_citation(self, gold, nda_chunks):
        """A clause citing a position the document does not have must fail.

        This is the 'what does Section 99 say' failure mode. evaluation_v3.py
        cannot see it: the clause is non-empty, the contract type is right, and
        no blocklisted phrase appears.
        """
        fabricated = [
            {
                "clause_type": "liability",
                "source_position": 99,
                "text": "Section 99 imposes a penalty of $1,000,000 on early termination.",
            }
        ]
        scored = ef.score_grounding(gold["fixtures"][0]["clauses"], fabricated, nda_chunks)
        assert scored["grounded"] == 0
        assert scored["rate"] == 0.0
        assert [f["kind"] for f in scored["failures"]] == ["dangling_citation"]

    def test_rejects_text_that_is_not_at_the_cited_position(self, gold, nda_chunks):
        """Valid position, invented quote: the citation does not support the clause."""
        misquoted = [
            {
                "clause_type": "liability",
                "source_position": 1,
                "text": "The Receiving Party shall pay liquidated damages of $50,000.",
            }
        ]
        scored = ef.score_grounding(gold["fixtures"][0]["clauses"], misquoted, nda_chunks)
        assert scored["grounded"] == 0
        assert [f["kind"] for f in scored["failures"]] == ["text_mismatch"]

    def test_rejects_clause_with_no_citation(self, gold, nda_chunks):
        uncited = [{"clause_type": "confidentiality", "source_position": None, "text": "x"}]
        scored = ef.score_grounding(gold["fixtures"][0]["clauses"], uncited, nda_chunks)
        assert [f["kind"] for f in scored["failures"]] == ["missing_citation"]

    def test_rejects_citation_to_the_wrong_passage(self, nda_chunks):
        """Quote is real and at the cited position, but that is not the gold passage.

        Guards against gold/document drift: the quote verifies against the
        document while the anchor verifies the position is the one the gold
        label is about.
        """
        synthetic_gold = [
            {
                "source_position": 1,
                "gold_clause_type": "confidentiality",
                "must_contain": "laws of the State of California",
            }
        ]
        quoted_from_pos_1 = nda_chunks[1]["text"]
        scored = ef.score_grounding(
            synthetic_gold,
            [{"clause_type": "confidentiality", "source_position": 1, "text": quoted_from_pos_1}],
            nda_chunks,
        )
        assert [f["kind"] for f in scored["failures"]] == ["wrong_passage"]


class TestClauseTypeScoring:
    GOLD = [
        {"source_position": 0, "gold_clause_type": "confidentiality"},
        {"source_position": 1, "gold_clause_type": "liability"},
    ]

    def test_all_correct(self):
        reported = [
            {"source_position": 0, "clause_type": "confidentiality"},
            {"source_position": 1, "clause_type": "liability"},
        ]
        s = ef.score_clause_types(self.GOLD, reported)
        assert (s["tp"], s["fp"], s["fn"]) == (2, 0, 0)
        assert s["precision"] == 1.0 and s["recall"] == 1.0 and s["f1"] == 1.0

    def test_wrong_label_costs_both_precision_and_recall(self):
        reported = [
            {"source_position": 0, "clause_type": "confidentiality"},
            {"source_position": 1, "clause_type": "payment"},
        ]
        s = ef.score_clause_types(self.GOLD, reported)
        assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 1)
        assert s["precision"] == 0.5 and s["recall"] == 0.5
        assert [e["kind"] for e in s["errors"]] == ["wrong_clause_type"]

    def test_dropped_clause_costs_recall_only(self):
        """A clause removed by the 0.6 confidence filter is a recall failure."""
        reported = [{"source_position": 0, "clause_type": "confidentiality"}]
        s = ef.score_clause_types(self.GOLD, reported)
        assert (s["tp"], s["fp"], s["fn"]) == (1, 0, 1)
        assert s["precision"] == 1.0
        assert s["recall"] == 0.5
        assert [e["kind"] for e in s["errors"]] == ["clause_not_reported"]

    def test_extra_clause_costs_precision_only(self):
        reported = [
            {"source_position": 0, "clause_type": "confidentiality"},
            {"source_position": 1, "clause_type": "liability"},
            {"source_position": 7, "clause_type": "payment"},
        ]
        s = ef.score_clause_types(self.GOLD, reported)
        assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 0)
        assert s["recall"] == 1.0
        assert s["precision"] < 1.0
        assert [e["kind"] for e in s["errors"]] == ["unlabelled_position"]

    def test_acceptable_alternative_label_counts_as_correct(self):
        gold = [
            {
                "source_position": 0,
                "gold_clause_type": "termination",
                "acceptable": ["term_and_termination"],
            }
        ]
        s = ef.score_clause_types(
            gold, [{"source_position": 0, "clause_type": "term_and_termination"}]
        )
        assert (s["tp"], s["fp"], s["fn"]) == (1, 0, 0)


class TestIntentScoring:
    def test_match_and_mismatch(self):
        assert ef.score_intent("payment", {"intent": "payment"})["correct"] is True
        miss = ef.score_intent("payment", {"intent": "termination"})
        assert miss["correct"] is False
        assert miss["got"] == "termination"

    def test_missing_intent_is_not_silently_correct(self):
        assert ef.score_intent("payment", {})["correct"] is False


class TestEndToEnd:
    @pytest.fixture(scope="class")
    def report(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("factual_eval")
        return ef.evaluate(
            use_mock=True,
            log_dir=str(out / "traces"),
            report_dir=str(out / "reports"),
        )

    def test_gold_labels_are_consistent_with_fixtures(self, report):
        assert report["gold_integrity_problems"] == []

    def test_scores_do_not_regress_below_baseline(self, report):
        overall = report["overall"]
        assert overall["clause_typing"]["f1"] >= ef.BASELINE["clause_type_f1"]
        assert overall["grounding"]["rate"] >= ef.BASELINE["grounding_rate"]
        assert overall["intent"]["accuracy"] >= ef.BASELINE["intent_accuracy"]

    def test_measured_scores_are_exactly_what_is_documented(self, report):
        """Pin the numbers quoted in the README so they cannot drift unnoticed."""
        overall = report["overall"]
        assert overall["clause_typing"] == {
            "tp": 9,
            "fp": 4,
            "fn": 4,
            "precision": 0.692,
            "recall": 0.692,
            "f1": 0.692,
        }
        assert overall["grounding"] == {"checked": 13, "grounded": 13, "rate": 1.0}
        assert overall["intent"] == {"total": 7, "correct": 6, "accuracy": 0.857}

    def test_known_defects_are_still_reported(self, report):
        """The eval must keep naming the specific mislabels, not just a score."""
        errors = [
            (f["fixture"], e["source_position"], e["reported"], e["expected"])
            for f in report["fixtures"]
            for e in f["clause_typing"]["errors"]
            if e["kind"] == "wrong_clause_type"
        ]
        assert ("tests/fixtures/clean_vendor.txt", 4, "payment", "liability") in errors
        assert ("tests/fixtures/clean_nda.txt", 2, "confidentiality", "exclusions") in errors
        assert (
            "tests/fixtures/clean_nda.txt",
            3,
            "term_and_termination",
            "return_of_materials",
        ) in errors

    def test_known_intent_failure_is_still_reported(self, report):
        misses = [
            i
            for f in report["fixtures"]
            for i in f["intent"]
            if not i["correct"]
        ]
        assert len(misses) == 1
        assert misses[0]["expected"] == "payment"
        assert misses[0]["got"] == "termination"

    def test_strict_mode_passes_at_baseline(self, report):
        assert ef.check_regressions(report) == []

    def test_strict_mode_fails_on_a_regression(self, report):
        degraded = json.loads(json.dumps(report))
        degraded["overall"]["grounding"]["rate"] = 0.5
        regressions = ef.check_regressions(degraded)
        assert any("grounding_rate" in r for r in regressions)
