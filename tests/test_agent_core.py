from __future__ import annotations

from unittest.mock import patch

import pytest

from legal_intelligence_agent.agent_core import AgentCore, AgentError


class TestAnalyze:
    def test_analyze_returns_risk_levels(self, agent_core, sample_nda_chunks):
        """test_analyze_returns_risk_levels ensures each analyzed clause has a valid risk level."""
        results = agent_core.analyze(sample_nda_chunks[:4], "What are the confidentiality obligations?", "confidentiality")
        assert len(results) == 4
        for r in results:
            assert "risk_level" in r
            assert r["risk_level"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

    def test_analyze_marks_triggered_clauses(self, agent_core, sample_vendor_chunks):
        """test_analyze_marks_triggered_clauses marks at least one clause as triggered for matching intent."""
        results = agent_core.analyze(sample_vendor_chunks, "Can I terminate this agreement?", "termination")
        assert any(r.get("triggered") is True for r in results)

    def test_analyze_includes_source_position(self, agent_core, sample_nda_chunks):
        """test_analyze_includes_source_position preserves source position from input chunks."""
        results = agent_core.analyze(sample_nda_chunks[:3], "Analyze risk", "liability")
        for original, analyzed in zip(sample_nda_chunks[:3], results):
            assert "source_position" in analyzed
            assert analyzed["source_position"] == original["position"]

    def test_analyze_empty_chunks_raises_error(self, agent_core):
        """test_analyze_empty_chunks_raises_error raises AgentError when chunks are empty."""
        with pytest.raises(AgentError):
            agent_core.analyze([], "Anything", "liability")


class TestTranslate:
    def test_translate_adds_plain_english(self, agent_core, sample_nda_chunks):
        """test_translate_adds_plain_english adds non-empty plain_english text for each clause."""
        analyzed = agent_core.analyze(sample_nda_chunks[:3], "What are obligations?", "confidentiality")
        translated = agent_core.translate(analyzed)
        for t in translated:
            assert "plain_english" in t
            assert isinstance(t["plain_english"], str)
            assert t["plain_english"].strip()

    def test_translate_removes_jargon(self, agent_core):
        """test_translate_removes_jargon removes prohibited legal jargon words."""
        analyzed = [
            {
                "text": "Hereinafter the party shall act pursuant to this clause notwithstanding objections.",
                "clause_type": "general",
                "position": 0,
                "char_count": 88,
                "risk_level": "LOW",
                "legal_meaning": "Meaning",
                "triggered": False,
                "source_position": 0,
                "confidence": 0.9,
            }
        ]
        translated = agent_core.translate(analyzed)
        plain = translated[0]["plain_english"].lower()
        assert "hereinafter" not in plain
        assert "pursuant" not in plain
        assert "notwithstanding" not in plain

    def test_translate_preserves_source_position(self, agent_core, sample_nda_chunks):
        """test_translate_preserves_source_position keeps source_position after translation."""
        analyzed = agent_core.analyze(sample_nda_chunks[:3], "Explain clauses", "confidentiality")
        translated = agent_core.translate(analyzed)
        for a, t in zip(analyzed, translated):
            assert t["source_position"] == a["source_position"]


class TestGapCheck:
    def test_gap_check_finds_missing_clauses(self, agent_core, sample_nda_chunks):
        """test_gap_check_finds_missing_clauses finds missing NDA template clauses."""
        # sample_nda_chunks deliberately has no governing_law clause type.
        analyzed = agent_core.analyze(sample_nda_chunks, "NDA risks", "confidentiality")
        missing = agent_core.gap_check(analyzed, "NDA")
        assert missing != []
        assert any(m["severity"] in {"CRITICAL", "HIGH", "MEDIUM"} for m in missing)

    def test_gap_check_returns_correct_structure(self, agent_core, sample_nda_chunks):
        """test_gap_check_returns_correct_structure validates missing clause dict schema."""
        analyzed = agent_core.analyze(sample_nda_chunks, "NDA risks", "confidentiality")
        missing = agent_core.gap_check(analyzed, "NDA")
        for m in missing:
            assert set(m.keys()) == {"clause_type", "severity", "description"}

    def test_gap_check_complete_contract_returns_empty(self, agent_core):
        """test_gap_check_complete_contract_returns_empty returns [] for complete NDA clause coverage."""
        complete_nda = [
            {"text": "Keep information confidential.", "clause_type": "confidentiality", "position": 0, "char_count": 28},
            {"text": "Public information is excluded.", "clause_type": "exclusions", "position": 1, "char_count": 31},
            {"text": "Return materials on request.", "clause_type": "return_of_materials", "position": 2, "char_count": 28},
            {"text": "Term is two years.", "clause_type": "term", "position": 3, "char_count": 18},
            {"text": "California law governs.", "clause_type": "governing_law", "position": 4, "char_count": 23},
        ]
        analyzed = agent_core.analyze(complete_nda, "Review NDA", "confidentiality")
        missing = agent_core.gap_check(analyzed, "NDA")
        assert missing == []


class TestRunAndValidate:
    def test_run_returns_complete_output(self, agent_core, sample_nda_chunks):
        """test_run_returns_complete_output returns required top-level keys."""
        result = agent_core.run(
            query="What are my confidentiality obligations?",
            intent="confidentiality",
            chunks=sample_nda_chunks,
            contract_type="NDA",
        )
        required = {
            "query",
            "intent",
            "contract_type",
            "analyzed_clauses",
            "missing_clauses",
            "summary",
            "total_clauses_found",
            "total_missing",
        }
        assert required.issubset(result.keys())

    def test_run_counts_are_correct(self, agent_core, sample_nda_chunks):
        """test_run_counts_are_correct keeps output counts aligned with returned lists."""
        result = agent_core.run("Analyze NDA risk", "confidentiality", sample_nda_chunks, contract_type="NDA")
        assert result["total_clauses_found"] == len(result["analyzed_clauses"])
        assert result["total_missing"] == len(result["missing_clauses"])
        assert result["critical_count"] == sum(
            1 for c in result["analyzed_clauses"] if c["risk_level"] == "CRITICAL"
        )

    def test_validate_removes_low_confidence(self, agent_core):
        """test_validate_removes_low_confidence filters out clauses below confidence threshold."""
        analyzed = [
            {
                "text": "A",
                "clause_type": "general",
                "position": 0,
                "char_count": 1,
                "risk_level": "LOW",
                "legal_meaning": "x",
                "triggered": False,
                "source_position": 0,
                "confidence": 0.3,
            },
            {
                "text": "B",
                "clause_type": "general",
                "position": 1,
                "char_count": 1,
                "risk_level": "HIGH",
                "legal_meaning": "y",
                "triggered": True,
                "source_position": 1,
                "confidence": 0.9,
            },
        ]
        translated = [dict(a, plain_english=f"Clause {a['text']} explanation.") for a in analyzed]
        with patch.object(
            agent_core.llm_client,
            "validate",
            return_value={"analyzed_clauses": translated, "missing_clauses": [], "summary": "Short summary"},
        ):
            validated = agent_core.validate(analyzed, translated, [])
        assert len(validated["analyzed_clauses"]) == 1
        assert validated["analyzed_clauses"][0]["confidence"] >= 0.6

    def test_run_raises_error_with_no_llm_client(self, sample_nda_chunks):
        """test_run_raises_error_with_no_llm_client raises AgentError when llm_client is None."""
        core = AgentCore(llm_client=None)
        with pytest.raises(AgentError):
            core.run("Review this", "confidentiality", sample_nda_chunks, "NDA")

    def test_run_raises_error_with_empty_chunks(self, agent_core):
        """test_run_raises_error_with_empty_chunks raises AgentError on empty chunk input."""
        with pytest.raises(AgentError):
            agent_core.run("Review this", "confidentiality", [], "NDA")

    def test_summary_is_plain_english(self, agent_core, sample_nda_chunks):
        """test_summary_is_plain_english ensures summary is readable and free of legal jargon."""
        result = agent_core.run("Explain this NDA", "confidentiality", sample_nda_chunks, "NDA")
        summary = result["summary"]
        assert isinstance(summary, str)
        assert len(summary) > 20
        low = summary.lower()
        assert "hereinafter" not in low
        assert "pursuant" not in low
        assert "notwithstanding" not in low
        assert "whereas" not in low

