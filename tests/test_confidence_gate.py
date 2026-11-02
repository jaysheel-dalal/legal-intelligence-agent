"""Tests for the 0.6 confidence gate and the display-only estimate.

The README lists "Confidence threshold -- results with confidence below 0.6 are
removed" as an anti-hallucination measure. It only means something if a clause
the model never scored is also removed. Four sites implemented the rule and
three of them substituted a number for a missing one, so an unscored clause
could reach the user labelled up to 97% confident.

These tests pin the invariant: a confidence that was not reported by the model
can never satisfy the gate, no matter which layer is asked.

app.py itself is not importable under pytest (it calls st.set_page_config at
module scope), which is precisely why the formatting logic now lives in
legal_intelligence_agent.confidence -- confidence_for_display is the function
app.py calls, tested here directly.
"""

import pytest

from legal_intelligence_agent.agent_core import AgentCore
from legal_intelligence_agent.confidence import (
    CONFIDENCE_THRESHOLD,
    confidence_for_display,
    drop_counts,
    estimated_confidence,
    model_confidence,
    partition_by_threshold,
    passes_threshold,
)
from main_v3 import ClaudeLLMClient, MockLLMClient


RISK_LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN", None]


class TestModelConfidence:
    def test_reports_a_real_value(self):
        assert model_confidence({"confidence": 0.83}) == 0.83

    def test_parses_a_numeric_string(self):
        assert model_confidence({"confidence": "0.83"}) == 0.83

    def test_absent_is_none_not_a_number(self):
        assert model_confidence({}) is None
        assert model_confidence({"confidence": None}) is None

    def test_unparseable_is_none(self):
        assert model_confidence({"confidence": "high"}) is None
        assert model_confidence({"confidence": []}) is None

    def test_bool_is_not_a_confidence(self):
        """float(True) is 1.0, which would read as 100% confident."""
        assert model_confidence({"confidence": True}) is None
        assert model_confidence({"confidence": False}) is None


class TestPassesThreshold:
    def test_above_threshold_passes(self):
        assert passes_threshold({"confidence": 0.9}) is True

    def test_exactly_at_threshold_passes(self):
        assert passes_threshold({"confidence": CONFIDENCE_THRESHOLD}) is True

    def test_just_below_threshold_fails(self):
        assert passes_threshold({"confidence": 0.59}) is False

    def test_missing_confidence_fails(self):
        """The core fix: absent is not 0.7, not 1.0, and not a pass."""
        assert passes_threshold({}) is False
        assert passes_threshold({"risk_level": "CRITICAL", "triggered": True}) is False

    def test_unparseable_confidence_fails(self):
        assert passes_threshold({"confidence": "very high"}) is False


class TestEstimateCannotPassTheGate:
    @pytest.mark.parametrize("risk", RISK_LEVELS)
    @pytest.mark.parametrize("triggered", [True, False])
    def test_every_estimate_would_clear_the_threshold(self, risk, triggered):
        """Why the estimate must not be stored as `confidence`.

        The derived scale is 0.62 to 0.97, so its floor sits above the 0.6
        gate: every possible estimate would pass. That is not a bug in the
        scale, it is the reason the value has to be kept out of the key the
        gate reads.
        """
        assert estimated_confidence(risk, triggered) >= CONFIDENCE_THRESHOLD

    @pytest.mark.parametrize("risk", RISK_LEVELS)
    @pytest.mark.parametrize("triggered", [True, False])
    def test_formatted_clause_never_passes_the_gate_on_an_estimate(self, risk, triggered):
        """The regression test for the app.py fabrication.

        A clause the model did not score gets a display number, but the clause
        that comes out must still fail the gate. Previously the estimate was
        written to "confidence" itself, so this would have passed.
        """
        unscored = {"risk_level": risk, "triggered": triggered}
        conf, display, is_estimated = confidence_for_display(unscored)

        assert conf is None
        assert is_estimated is True
        assert display >= CONFIDENCE_THRESHOLD

        formatted = {
            "confidence": conf,
            "display_confidence": display,
            "confidence_estimated": is_estimated,
        }
        assert passes_threshold(formatted) is False

    def test_reported_confidence_is_passed_through_untouched(self):
        conf, display, is_estimated = confidence_for_display({"confidence": 0.83})
        assert (conf, display, is_estimated) == (0.83, 0.83, False)

    @pytest.mark.parametrize("value", [0.7, 0.8, 0.65])
    def test_former_sentinel_values_are_no_longer_discarded(self, value):
        """app.py treated {0.7, 0.8, 0.65} as sentinels and overwrote them.

        A genuine model confidence of 0.7 was replaced by a derived number.
        The 0.7 it was chasing came from agent_core.analyze and is fixed there.
        """
        conf, display, is_estimated = confidence_for_display({"confidence": value})
        assert conf == value
        assert display == value
        assert is_estimated is False

    def test_low_risk_untriggered_estimate_sits_just_above_the_gate(self):
        """Documents how narrow the old bypass was: 0.62 vs a 0.6 gate."""
        assert estimated_confidence("LOW", triggered=False) == 0.62


class _UnscoredLLM:
    """Model that returns no confidence field at all."""

    def analyze(self, chunks, query, intent):
        return [
            {"risk_level": "CRITICAL", "legal_meaning": "m", "triggered": True}
            for _ in chunks
        ]

    def translate(self, clause):
        return "Plain explanation of the clause."

    def gap_check(self, analyzed, contract_type):
        return []

    def validate(self, analyzed, translated, gaps):
        return {
            "analyzed_clauses": translated,
            "missing_clauses": gaps,
            "summary": "s",
        }


class TestGatesRejectUnscoredClauses:
    CHUNKS = [
        {"text": "Either party may terminate on notice.", "clause_type": "termination", "position": 0}
    ]

    def test_agent_core_analyze_leaves_confidence_unset(self):
        """Was float(item.get("confidence", 0.7)), which cleared the gate."""
        analyzed = AgentCore(llm_client=_UnscoredLLM()).analyze(
            self.CHUNKS, "when can this end?", "termination"
        )
        assert analyzed[0]["confidence"] is None

    def test_agent_core_drops_the_unscored_clause(self):
        result = AgentCore(llm_client=_UnscoredLLM()).run(
            query="when can this end?",
            intent="termination",
            chunks=self.CHUNKS,
            contract_type="NDA",
        )
        assert result["analyzed_clauses"] == []
        assert result["total_clauses_found"] == 0

    def test_agent_core_keeps_a_scored_clause(self):
        """The gate must still let real results through."""

        class Scored(_UnscoredLLM):
            def analyze(self, chunks, query, intent):
                return [
                    {
                        "risk_level": "HIGH",
                        "legal_meaning": "m",
                        "triggered": True,
                        "confidence": 0.81,
                    }
                    for _ in chunks
                ]

        result = AgentCore(llm_client=Scored()).run(
            query="when can this end?",
            intent="termination",
            chunks=self.CHUNKS,
            contract_type="NDA",
        )
        assert len(result["analyzed_clauses"]) == 1
        assert result["analyzed_clauses"][0]["confidence"] == 0.81

    def test_mock_client_validate_drops_unscored(self):
        unscored = [{"source_position": 0, "risk_level": "CRITICAL", "text": "t"}]
        out = MockLLMClient().validate(unscored, unscored, [])
        assert out["analyzed_clauses"] == []
        assert out["total_clauses_found"] == 0

    def test_claude_client_validate_drops_unscored(self):
        """Was c.get("confidence", 1.0) >= 0.6: missing meant full confidence.

        Called unbound with a dummy self: validate touches no instance state,
        and constructing ClaudeLLMClient would need the SDK and an API key,
        which would make this untestable in CI.
        """
        unscored = [{"source_position": 0, "risk_level": "CRITICAL", "text": "t"}]
        out = ClaudeLLMClient.validate(None, unscored, unscored, [])
        assert out["analyzed_clauses"] == []
        assert out["total_clauses_found"] == 0

    def test_claude_client_validate_keeps_scored(self):
        scored = [
            {"source_position": 0, "risk_level": "HIGH", "text": "t", "confidence": 0.77}
        ]
        out = ClaudeLLMClient.validate(None, scored, scored, [])
        assert out["total_clauses_found"] == 1

    def test_all_gates_agree_on_a_missing_confidence(self):
        """The bug was four sites disagreeing. Pin that they now agree."""
        unscored = [{"source_position": 0, "risk_level": "CRITICAL", "text": "t"}]

        assert passes_threshold(unscored[0]) is False
        assert MockLLMClient().validate(unscored, unscored, [])["total_clauses_found"] == 0
        assert ClaudeLLMClient.validate(None, unscored, unscored, [])[
            "total_clauses_found"
        ] == 0
        assert AgentCore(llm_client=_UnscoredLLM()).validate(
            unscored, unscored, []
        )["analyzed_clauses"] == []


class TestGateObservability:
    """The gate reports what it dropped and why.

    It used to drop silently, so a thin result could not be told apart from a
    model that declined to score anything. The two causes are counted
    separately because they mean different things.
    """

    CLAUSES = [
        {"confidence": 0.91, "source_position": 0},
        {"confidence": 0.42, "source_position": 1},
        {"source_position": 2},
        {"confidence": "n/a", "source_position": 3},
        {"confidence": CONFIDENCE_THRESHOLD, "source_position": 4},
    ]

    def test_partition_separates_the_two_drop_causes(self):
        kept, missing, low = partition_by_threshold(self.CLAUSES)
        assert [c["source_position"] for c in kept] == [0, 4]
        assert [c["source_position"] for c in missing] == [2, 3]
        assert [c["source_position"] for c in low] == [1]

    def test_partition_handles_no_clauses(self):
        assert partition_by_threshold([]) == ([], [], [])
        assert partition_by_threshold(None) == ([], [], [])

    def test_drop_counts_shape(self):
        _, missing, low = partition_by_threshold(self.CLAUSES)
        assert drop_counts(missing, low) == {
            "dropped_missing_confidence": 2,
            "dropped_low_confidence": 1,
        }

    def test_mock_client_validate_reports_drop_counts(self):
        out = MockLLMClient().validate(self.CLAUSES, self.CLAUSES, [])
        assert out["dropped_missing_confidence"] == 2
        assert out["dropped_low_confidence"] == 1
        assert out["total_clauses_found"] == 2

    def test_claude_client_validate_reports_drop_counts(self):
        out = ClaudeLLMClient.validate(None, self.CLAUSES, self.CLAUSES, [])
        assert out["dropped_missing_confidence"] == 2
        assert out["dropped_low_confidence"] == 1

    def test_agent_core_validate_reports_drop_counts(self):
        out = AgentCore(llm_client=_UnscoredLLM()).validate(
            self.CLAUSES, self.CLAUSES, []
        )
        assert out["dropped_missing_confidence"] == 2
        assert out["dropped_low_confidence"] == 1

    def test_orchestrator_surfaces_the_counters_into_the_trace(self, tmp_path):
        """The counters must reach the trace, not stop at the report agent."""
        from agents.orchestrator import Orchestrator

        result = Orchestrator(
            llm_client=MockLLMClient(),
            log_dir=str(tmp_path),
            output_format="dict",
        ).run(
            query="What are my confidentiality obligations?",
            file_path="tests/fixtures/clean_nda.txt",
        )
        assert "dropped_missing_confidence" in result
        assert "dropped_low_confidence" in result

    def test_mock_client_never_exercises_the_missing_path(self):
        """Why the mock cannot answer "how many were dropped for missing".

        MockLLMClient.analyze always sets a confidence, so a mock run reports
        zero missing-confidence drops by construction. Only a live model can
        omit the field, which is the case the gate exists for.
        """
        chunks = [{"text": "Confidential Information.", "clause_type": "confidentiality", "position": 0}]
        analyzed = MockLLMClient().analyze(chunks, "confidentiality?", "confidentiality")
        assert all(model_confidence(a) is not None for a in analyzed)
