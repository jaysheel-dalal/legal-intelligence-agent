import copy

import pytest

from legal_intelligence_agent.output_report import (
    DISCLAIMER,
    OutputReport,
    ReportError,
)


class TestDictOutput:
    def test_dict_has_required_keys(self, sample_agent_result):
        report = OutputReport()
        output = report.generate(sample_agent_result, format="dict")

        assert set(output.keys()) == {
            "metadata",
            "clauses",
            "missing_clauses",
            "summary",
            "disclaimer",
        }

    def test_clauses_sorted_by_risk(self, sample_agent_result):
        report = OutputReport()
        output = report.generate(sample_agent_result, format="dict")

        clauses = output["clauses"]
        assert clauses[0]["risk_level"] == "CRITICAL"
        assert clauses[-1]["risk_level"] == "LOW"

        assert [c["risk_level"] for c in clauses] == [
            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW",
        ]

    def test_missing_clauses_sorted_by_severity(self, sample_agent_result):
        report = OutputReport()
        output = report.generate(sample_agent_result, format="dict")

        missing = output["missing_clauses"]
        assert missing[0]["severity"] == "CRITICAL"
        assert missing[1]["severity"] == "HIGH"

    def test_disclaimer_is_correct(self, sample_agent_result):
        report = OutputReport()
        output = report.generate(sample_agent_result, format="dict")

        assert output["disclaimer"] == DISCLAIMER

    def test_metadata_counts_match(self, sample_agent_result):
        report = OutputReport()
        output = report.generate(sample_agent_result, format="dict")

        metadata = output["metadata"]
        assert metadata["total_clauses_found"] == len(output["clauses"])
        assert metadata["critical_count"] == sum(
            1 for c in output["clauses"] if c.get("risk_level") == "CRITICAL"
        )


class TestTextOutput:
    def test_text_output_is_string(self, sample_agent_result):
        report = OutputReport()
        output = report.generate(sample_agent_result, format="text")
        assert isinstance(output, str)

    def test_text_has_all_sections(self, sample_agent_result):
        report = OutputReport()
        output = report.generate(sample_agent_result, format="text")

        assert "SUMMARY" in output
        assert "CRITICAL CLAUSES" in output
        assert "MISSING CLAUSES" in output
        assert "DISCLAIMER" in output

    def test_text_shows_plain_english(self, sample_agent_result):
        report = OutputReport()
        output = report.generate(sample_agent_result, format="text")

        plain = sample_agent_result["analyzed_clauses"][0]["plain_english"]
        assert plain in output

    def test_text_shows_source_position(self, sample_agent_result):
        report = OutputReport()
        output = report.generate(sample_agent_result, format="text")

        assert "Source: position" in output

    def test_text_disclaimer_present(self, sample_agent_result):
        report = OutputReport()
        output = report.generate(sample_agent_result, format="text")

        assert "legal advice" in output.lower()
        assert "lawyer" in output.lower()

    def test_text_skips_empty_risk_sections(self, sample_agent_result):
        modified = copy.deepcopy(sample_agent_result)
        modified["analyzed_clauses"] = [
            c for c in modified["analyzed_clauses"] if c.get("risk_level") != "LOW"
        ]

        report = OutputReport()
        output = report.generate(modified, format="text")

        assert "LOW RISK CLAUSES" not in output


class TestMarkdownOutput:
    def test_markdown_is_string(self, sample_agent_result):
        report = OutputReport()
        output = report.generate(sample_agent_result, format="markdown")
        assert isinstance(output, str)

    def test_markdown_has_headers(self, sample_agent_result):
        report = OutputReport()
        output = report.generate(sample_agent_result, format="markdown")
        assert "## " in output

    def test_markdown_has_critical_emoji(self, sample_agent_result):
        report = OutputReport()
        output = report.generate(sample_agent_result, format="markdown")
        assert "🔴" in output

    def test_markdown_has_high_emoji(self, sample_agent_result):
        report = OutputReport()
        output = report.generate(sample_agent_result, format="markdown")
        assert "🟠" in output

    def test_markdown_disclaimer_is_blockquote(self, sample_agent_result):
        report = OutputReport()
        output = report.generate(sample_agent_result, format="markdown")
        assert "> This is legal information" in output

    def test_markdown_plain_english_is_bold(self, sample_agent_result):
        report = OutputReport()
        output = report.generate(sample_agent_result, format="markdown")
        assert "**" in output


class TestEdgeCases:
    def test_none_result_raises_report_error(self):
        report = OutputReport()
        with pytest.raises(ReportError):
            report.generate(None, format="dict")

    def test_invalid_format_raises_report_error(self, sample_agent_result):
        report = OutputReport()
        with pytest.raises(ReportError):
            report.generate(sample_agent_result, format="xml")

    def test_missing_required_key_raises_error(self, sample_agent_result):
        report = OutputReport()
        modified = copy.deepcopy(sample_agent_result)
        modified.pop("analyzed_clauses", None)

        with pytest.raises(ReportError):
            report.generate(modified, format="dict")

    def test_empty_clauses_generates_cleanly(self, sample_agent_result):
        report = OutputReport()
        modified = copy.deepcopy(sample_agent_result)
        modified["analyzed_clauses"] = []

        output = report.generate(modified, format="dict")
        assert output["clauses"] == []
        assert output["disclaimer"] == DISCLAIMER

    def test_no_missing_clauses_shows_none_found(self, sample_agent_result):
        report = OutputReport()
        modified = copy.deepcopy(sample_agent_result)
        modified["missing_clauses"] = []

        text_output = report.generate(modified, format="text")
        assert "None found" in text_output

        markdown_output = report.generate(modified, format="markdown")
        assert "> None found" in markdown_output

    def test_all_formats_produce_output(self, sample_agent_result):
        report = OutputReport()

        for fmt in ["dict", "text", "markdown"]:
            output = report.generate(sample_agent_result, format=fmt)
            assert isinstance(output, (dict, str))
            assert output

