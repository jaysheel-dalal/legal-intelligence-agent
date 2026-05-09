import pytest
from unittest.mock import patch

from legal_intelligence_agent.document_parser import ParseError, UnsupportedFileTypeError


class TestHappyPath:
    def test_parse_clean_nda(self, document_parser, clean_nda_path):
        """test_parse_clean_nda parses clean NDA and detects confidentiality content."""
        result = document_parser.parse_document(document_path=clean_nda_path)
        chunks = result["chunks"]

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert any("confidential" in c["text"].lower() for c in chunks)
        assert result["contract_type"] == "NDA"

    def test_parse_clean_vendor_agreement(self, document_parser, clean_vendor_path):
        """test_parse_clean_vendor_agreement parses vendor agreement and finds payment/liability text."""
        result = document_parser.parse_document(document_path=clean_vendor_path)
        chunks = result["chunks"]

        assert result["contract_type"] in {"vendor", "supplier"}
        assert len(chunks) > 0
        assert any(("payment" in c["text"].lower() or "liability" in c["text"].lower()) for c in chunks)
        assert all(isinstance(c["text"], str) and c["text"].strip() for c in chunks)

    def test_chunk_structure(self, document_parser, clean_vendor_path):
        """test_chunk_structure validates chunk schema and sequential positions."""
        result = document_parser.parse_document(document_path=clean_vendor_path)
        chunks = result["chunks"]

        positions = []
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert set(chunk.keys()) == {"text", "clause_type", "position", "char_count"}
            assert isinstance(chunk["text"], str)
            assert isinstance(chunk["clause_type"], str)
            assert isinstance(chunk["position"], int)
            assert isinstance(chunk["char_count"], int)
            assert chunk["char_count"] == len(chunk["text"])
            positions.append(chunk["position"])

        assert positions == list(range(len(chunks)))

    @pytest.mark.parametrize(
        "document_text, expected",
        [
            (
                "NON-DISCLOSURE AGREEMENT\n\nConfidential Information means all nonpublic information.\n\nThe Receiving Party shall keep it confidential.",
                "NDA",
            ),
            (
                "VENDOR SERVICES AGREEMENT\n\nSupplier shall provide services. Fees and payment are due within thirty days. Limitation of liability applies.",
                "vendor_or_supplier",
            ),
            (
                "EMPLOYMENT AGREEMENT\n\nThis Employment Agreement is between the Employer and the Employee.\n\nEmployee shall comply with restrictive covenants and non-solicitation obligations.",
                "employment",
            ),
            (
                "SAAS SUBSCRIPTION AGREEMENT\n\nCustomer receives a subscription to the SaaS service. Service levels and support apply.",
                "SaaS",
            ),
        ],
    )
    def test_contract_type_detection(self, document_parser, document_text, expected):
        """test_contract_type_detection detects contract_type for multiple sample documents."""
        result = document_parser.parse_document(document_text=document_text)

        if expected == "vendor_or_supplier":
            assert result["contract_type"] in {"vendor", "supplier"}
        else:
            assert result["contract_type"] == expected


class TestEdgeCases:
    def test_empty_file_returns_error(self, document_parser, empty_pdf_path):
        """test_empty_file_returns_error raises ParseError for an empty PDF."""
        with pytest.raises(ParseError) as excinfo:
            document_parser.parse_document(document_path=empty_pdf_path)

        msg = str(excinfo.value).lower()
        assert "empty" in msg or "no content" in msg

    def test_corrupt_docx_returns_error(self, document_parser, corrupt_docx_path):
        """test_corrupt_docx_returns_error raises ParseError with a meaningful message."""
        with pytest.raises(ParseError) as excinfo:
            document_parser.parse_document(document_path=corrupt_docx_path)

        msg = str(excinfo.value).lower()
        assert "corrupt" in msg or "malformed" in msg or "docx" in msg

    def test_whitespace_only_document(self, document_parser):
        """test_whitespace_only_document handles whitespace-only input without hanging."""
        whitespace_text = " \n\t  \n"
        try:
            result = document_parser.parse_document(document_text=whitespace_text)
            assert "chunks" in result
            assert result["chunks"] == [] or len(result["chunks"]) == 0
        except ParseError as exc:
            msg = str(exc).lower()
            assert "empty" in msg or "no content" in msg

    def test_unsupported_file_type(self, document_parser, tmp_path):
        """test_unsupported_file_type raises UnsupportedFileTypeError for unsupported extensions."""
        jpg_path = tmp_path / "image.jpg"
        jpg_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

        with pytest.raises(UnsupportedFileTypeError) as excinfo:
            document_parser.parse_document(document_path=jpg_path)

        msg = str(excinfo.value).lower()
        assert "jpg" in msg or ".jpg" in msg


class TestVagueQuery:
    def test_vague_query_triggers_clarification(self, document_parser):
        """test_vague_query_triggers_clarification flags clarification_needed for vague requests."""
        with patch.object(
            document_parser.llm_client,
            "classify_query",
            return_value={
                "clarification_needed": True,
                "suggested_topics": ["Confidentiality", "Payment", "Termination"],
            },
        ):
            result = document_parser.parse(query="legal stuff")

        assert result["clarification_needed"] is True
        assert isinstance(result["suggested_topics"], list)
        assert len(result["suggested_topics"]) >= 2
        assert result["chunks"] == [] or len(result["chunks"]) == 0

    def test_single_word_query(self, document_parser):
        """test_single_word_query flags clarification_needed and returns suggested topics."""
        with patch.object(
            document_parser.llm_client,
            "classify_query",
            return_value={
                "clarification_needed": True,
                "suggested_topics": ["Liability", "Confidentiality"],
            },
        ):
            result = document_parser.parse(query="contract")

        assert result["clarification_needed"] is True
        assert isinstance(result["suggested_topics"], list)
        assert len(result["suggested_topics"]) >= 1

    def test_specific_query_does_not_trigger_clarification(self, document_parser):
        """test_specific_query_does_not_trigger_clarification returns an intent label."""
        with patch.object(
            document_parser.llm_client,
            "classify_query",
            return_value={
                "clarification_needed": False,
                "intent": "termination",
                "suggested_topics": [],
            },
        ):
            result = document_parser.parse(query="What happens if I terminate this vendor agreement early?")

        assert result["clarification_needed"] is False
        assert result["intent"] in {"termination", "breach", "payment", "data", "IP", "employment", "liability"}


class TestAdversarial:
    def test_query_for_nonexistent_clause(self, document_parser, clean_nda_path):
        """test_query_for_nonexistent_clause returns clause_found False and a non-hallucinated message."""
        with patch.object(
            document_parser.llm_client,
            "classify_query",
            return_value={"clarification_needed": False, "intent": "payment", "suggested_topics": []},
        ):
            result = document_parser.parse(
                document_path=clean_nda_path,
                query="What is the payment schedule?",
            )

        assert result["clause_found"] is False
        assert result["message"] == "Clause not found in document"
        assert result["returned_clauses"] == [] or len(result["returned_clauses"]) == 0

    def test_query_returns_only_grounded_clauses(self, document_parser, clean_vendor_path):
        """test_query_returns_only_grounded_clauses does not invent clauses when retrieval is empty."""
        with patch.object(
            document_parser.llm_client,
            "classify_query",
            return_value={"clarification_needed": False, "intent": "IP", "suggested_topics": []},
        ), patch.object(document_parser.embedder, "retrieve", return_value=[]):
            result = document_parser.parse(
                document_path=clean_vendor_path,
                query="Who owns the intellectual property and IP ownership rights?",
            )

        returned = result["returned_clauses"]
        assert returned == [] or len(returned) == 0

        for clause in returned:
            assert "source_position" in clause
            assert isinstance(clause["source_position"], int)
            assert 0 <= clause["source_position"] < len(result["chunks"])

    def test_out_of_scope_legal_question(self, document_parser):
        """test_out_of_scope_legal_question flags out_of_scope and routes to knowledge-only mode."""
        result = document_parser.parse(query="How do I file for bankruptcy?")
        assert result["out_of_scope"] is True
        assert result["mode"] == "knowledge_only"
        assert result["chunks"] == [] or len(result["chunks"]) == 0

