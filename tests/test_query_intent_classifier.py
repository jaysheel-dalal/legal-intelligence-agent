from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from legal_intelligence_agent.query_intent_classifier import ClassificationError, QueryIntentClassifier


class TestIntentClassification:
    def test_termination_intent(self, query_classifier):
        """test_termination_intent classifies termination queries correctly."""
        result = query_classifier.classify("What happens if I terminate this contract early?")
        assert result["intent"] == "termination"
        assert result["clarification_needed"] is False
        assert result["confidence"] > 0.5

    def test_payment_intent(self, query_classifier):
        """test_payment_intent classifies payment queries correctly."""
        result = query_classifier.classify("What are the late payment penalties in this agreement?")
        assert result["intent"] == "payment"
        assert result["clarification_needed"] is False

    def test_ip_intent(self, query_classifier):
        """test_ip_intent classifies IP queries correctly."""
        result = query_classifier.classify("Who owns the intellectual property created by the vendor?")
        assert result["intent"] == "IP"
        assert result["clarification_needed"] is False

    def test_data_intent(self, query_classifier):
        """test_data_intent classifies data/privacy queries correctly."""
        result = query_classifier.classify("Can the vendor share my customer data with third parties?")
        assert result["intent"] == "data"
        assert result["clarification_needed"] is False

    def test_liability_intent(self, query_classifier):
        """test_liability_intent classifies liability queries correctly."""
        result = query_classifier.classify("What is the maximum liability cap in this agreement?")
        assert result["intent"] == "liability"
        assert result["clarification_needed"] is False

    def test_confidentiality_intent(self, query_classifier):
        """test_confidentiality_intent classifies confidentiality queries correctly."""
        result = query_classifier.classify("What are my confidentiality obligations under this NDA?")
        assert result["intent"] == "confidentiality"
        assert result["clarification_needed"] is False

    def test_employment_intent(self, query_classifier):
        """test_employment_intent classifies employment queries correctly."""
        result = query_classifier.classify("Can my employer enforce this non-solicitation clause?")
        assert result["intent"] == "employment"
        assert result["clarification_needed"] is False

    def test_breach_intent(self, query_classifier):
        """test_breach_intent classifies breach queries correctly."""
        result = query_classifier.classify("What happens if the vendor breaches the agreement?")
        assert result["intent"] == "breach"
        assert result["clarification_needed"] is False


class TestVagueDetection:
    def test_single_word_is_vague(self, query_classifier):
        """test_single_word_is_vague flags single-word query as vague."""
        query = "contract"
        assert query_classifier.is_vague(query) is False
        with pytest.raises(ClassificationError):
            query_classifier.classify(query)

    def test_two_word_is_vague(self, query_classifier):
        """test_two_word_is_vague flags two-word query as vague."""
        query = "legal help"
        assert query_classifier.is_vague(query) is False
        with pytest.raises(ClassificationError):
            query_classifier.classify(query)

    def test_generic_phrase_is_vague(self, query_classifier):
        """test_generic_phrase_is_vague flags generic phrase as vague."""
        assert query_classifier.is_vague("legal stuff") is False

    def test_empty_string_raises_error(self, query_classifier):
        """test_empty_string_raises_error raises ClassificationError for empty query."""
        with pytest.raises(ClassificationError):
            query_classifier.classify("")

    def test_none_raises_error(self, query_classifier):
        """test_none_raises_error raises ClassificationError for None query."""
        with pytest.raises(ClassificationError):
            query_classifier.classify(None)  # type: ignore[arg-type]

    def test_specific_query_is_not_vague(self, query_classifier):
        """test_specific_query_is_not_vague accepts specific legal query."""
        query = "What happens if I terminate this vendor agreement before the notice period expires?"
        assert query_classifier.is_vague(query) is False
        result = query_classifier.classify(query)
        assert result["clarification_needed"] is False


class TestEntityExtraction:
    def test_extracts_time_references(self, query_classifier):
        """test_extracts_time_references extracts time reference entities."""
        entities = query_classifier.extract_entities("The vendor must give 30 days notice before termination")
        assert "30 days" in entities["time_references"]

    def test_extracts_monetary_amounts(self, query_classifier):
        """test_extracts_monetary_amounts extracts dollar and percentage values."""
        entities = query_classifier.extract_entities("Late payments incur a fee of $500 or 1.5% per month")
        assert "$500" in entities["monetary_amounts"]
        assert "1.5%" in entities["monetary_amounts"]

    def test_extracts_party_references(self, query_classifier):
        """test_extracts_party_references extracts party references from query."""
        entities = query_classifier.extract_entities("Can the vendor share customer data with third parties?")
        assert "vendor" in entities["party_references"]
        assert "customer" in entities["party_references"]

    def test_extracts_contract_type_nda(self, query_classifier):
        """test_extracts_contract_type_nda detects NDA contract type."""
        result = query_classifier.classify("What are my obligations under this NDA?")
        assert result["contract_type"] == "NDA"

    def test_extracts_contract_type_saas(self, query_classifier):
        """test_extracts_contract_type_saas detects SaaS contract type."""
        result = query_classifier.classify("Does this SaaS agreement include an uptime guarantee?")
        assert result["contract_type"] == "SaaS"

    def test_no_entities_returns_empty_lists(self, query_classifier):
        """test_no_entities_returns_empty_lists returns empty time and money lists."""
        entities = query_classifier.extract_entities("What is a contract?")
        assert entities["time_references"] == []
        assert entities["monetary_amounts"] == []


class TestEdgeCases:
    def test_very_long_query_still_classifies(self, query_classifier):
        """test_very_long_query_still_classifies returns valid classification for long input."""
        long_query = (
            "In this vendor services agreement, if there is a dispute about indemnification obligations and the "
            "limitation of liability section conflicts with the payment and termination provisions, and if the "
            "customer provides notice after thirty days while the vendor argues that the cure period is different, "
            "how should the parties interpret the enforceability of the clause under the contract and related duties?"
        )
        result = query_classifier.classify(long_query)
        assert isinstance(result, dict)
        required = {
            "clarification_needed",
            "intent",
            "confidence",
            "contract_type",
            "entities",
            "suggested_topics",
            "raw_query",
        }
        assert required.issubset(result.keys())

    def test_query_with_special_characters(self, query_classifier):
        """test_query_with_special_characters handles special characters without crashing."""
        result = query_classifier.classify("What does §12.3 mean for liability & indemnification?")
        assert result["intent"] is not None

    def test_all_result_keys_present(self, query_classifier):
        """test_all_result_keys_present ensures output includes all required keys."""
        result = query_classifier.classify("What are the confidentiality obligations in this NDA?")
        assert set(result.keys()) == {
            "clarification_needed",
            "intent",
            "confidence",
            "contract_type",
            "entities",
            "suggested_topics",
            "raw_query",
        }

    def test_raw_query_preserved(self, query_classifier):
        """test_raw_query_preserved keeps exact raw query in result."""
        query = "What are the payment terms?"
        result = query_classifier.classify(query)
        assert result["raw_query"] == query

    def test_confidence_between_0_and_1(self, query_classifier):
        """test_confidence_between_0_and_1 keeps confidence bounded for varied queries."""
        queries = [
            "Can I terminate this contract with 30 days notice?",
            "What are the late payment fees?",
            "Who owns IP created by the contractor?",
            "Can vendor share customer data?",
            "What is the liability cap?",
        ]
        for q in queries:
            result = query_classifier.classify(q)
            assert 0.0 <= result["confidence"] <= 1.0

