from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_intelligence_agent.knowledge_base import KnowledgeBase, KnowledgeBaseError


class TestLoading:
    def test_loads_all_frameworks(self, repo_root: Path):
        """test_loads_all_frameworks loads 6 legal frameworks and validates schema keys."""
        kb = KnowledgeBase(knowledge_dir=str(repo_root / "knowledge"))
        frameworks = kb.list_frameworks()
        assert len(frameworks) == 6
        assert sorted(frameworks, key=str.lower) == sorted(frameworks, key=str.lower)

        for fw_name in frameworks:
            fw = kb.get_framework(fw_name)
            assert set(fw.keys()) == {"framework", "description", "applies_to", "obligations"}
            assert isinstance(fw["applies_to"], list)
            assert isinstance(fw["obligations"], list)

    def test_loads_all_contract_templates(self, repo_root: Path):
        """test_loads_all_contract_templates loads 6 contract templates and required clauses."""
        kb = KnowledgeBase(knowledge_dir=str(repo_root / "knowledge"))
        contract_types = kb.list_contract_types()
        assert len(contract_types) == 7

        for ct in contract_types:
            req = kb.get_required_clauses(ct)
            assert isinstance(req, list)
            assert len(req) > 0

    def test_missing_knowledge_dir_raises_error(self, tmp_path: Path):
        """test_missing_knowledge_dir_raises_error raises KnowledgeBaseError if knowledge_dir is missing."""
        missing = tmp_path / "does_not_exist"
        with pytest.raises(KnowledgeBaseError):
            KnowledgeBase(knowledge_dir=str(missing))

    def test_malformed_json_raises_error(self, tmp_path: Path):
        """test_malformed_json_raises_error raises KnowledgeBaseError on malformed JSON."""
        kb_dir = tmp_path / "knowledge"
        (kb_dir / "legal_frameworks").mkdir(parents=True)
        (kb_dir / "contract_templates").mkdir(parents=True)

        (kb_dir / "legal_frameworks" / "bad.json").write_text("{not valid json", encoding="utf-8")

        with pytest.raises(KnowledgeBaseError):
            KnowledgeBase(knowledge_dir=str(kb_dir))


class TestRequiredClauses:
    def test_nda_required_clauses(self, repo_root: Path):
        """test_nda_required_clauses ensures NDA required clauses include key template types."""
        kb = KnowledgeBase(knowledge_dir=str(repo_root / "knowledge"))
        req = kb.get_required_clauses("nda")
        clause_types = {c.get("clause_type") for c in req}
        for needed in {"confidentiality", "term", "governing_law", "return_of_materials", "no_license_grant"}:
            assert needed in clause_types

    def test_vendor_required_clauses(self, repo_root: Path):
        """test_vendor_required_clauses ensures vendor required clauses include payment, limitation_of_liability, termination, services, IP."""
        kb = KnowledgeBase(knowledge_dir=str(repo_root / "knowledge"))
        req = kb.get_required_clauses("vendor")
        clause_types = {c.get("clause_type") for c in req}
        for needed in {"payment", "limitation_of_liability", "termination", "services", "IP"}:
            assert needed in clause_types

    def test_saas_required_clauses(self, repo_root: Path):
        """test_saas_required_clauses ensures SaaS required clauses include subscription, service levels, data, payment, termination."""
        kb = KnowledgeBase(knowledge_dir=str(repo_root / "knowledge"))
        req = kb.get_required_clauses("saas")
        clause_types = {c.get("clause_type") for c in req}
        for needed in {"subscription_and_fees", "service_level_agreement", "data_security", "acceptable_use_policy"}:
            assert needed in clause_types

    def test_unknown_contract_type_defaults_to_nda(self, repo_root: Path):
        """test_unknown_contract_type_defaults_to_nda returns default NDA clauses for unknown contract_type."""
        kb = KnowledgeBase(knowledge_dir=str(repo_root / "knowledge"))
        # Unknown types now default to NDA
        clauses = kb.get_required_clauses(
            "unknown_type")
        assert len(clauses) > 0
        # Should return NDA clauses as default

    def test_case_insensitive_lookup(self, repo_root: Path):
        """test_case_insensitive_lookup returns identical NDA required clauses for NDA/nda/Nda inputs."""
        kb = KnowledgeBase(knowledge_dir=str(repo_root / "knowledge"))
        r1 = kb.get_required_clauses("NDA")
        r2 = kb.get_required_clauses("nda")
        r3 = kb.get_required_clauses("Nda")
        assert r1 == r2 == r3


class TestFrameworks:
    def test_get_ccpa_framework(self, repo_root: Path):
        """test_get_ccpa_framework validates CCPA framework shape and minimum obligations."""
        kb = KnowledgeBase(knowledge_dir=str(repo_root / "knowledge"))
        fw = kb.get_framework("ccpa")
        assert isinstance(fw["obligations"], list)
        assert len(fw["obligations"]) >= 3
        for ob in fw["obligations"]:
            assert set(ob.keys()) == {"id", "title", "description", "severity", "clause_types_affected"}

    def test_get_employment_framework(self, repo_root: Path):
        """test_get_employment_framework validates employment framework applies_to is non-empty."""
        kb = KnowledgeBase(knowledge_dir=str(repo_root / "knowledge"))
        fw = kb.get_framework("employment")
        assert isinstance(fw["applies_to"], list)
        assert len(fw["applies_to"]) > 0

    def test_unknown_framework_raises_error(self, repo_root: Path):
        """test_unknown_framework_raises_error raises KnowledgeBaseError for unknown law name."""
        kb = KnowledgeBase(knowledge_dir=str(repo_root / "knowledge"))
        with pytest.raises(KnowledgeBaseError):
            kb.get_framework("unknown_law")

    def test_framework_obligations_have_severity(self, repo_root: Path):
        """test_framework_obligations_have_severity asserts every obligation has a valid severity."""
        kb = KnowledgeBase(knowledge_dir=str(repo_root / "knowledge"))
        allowed = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        for fw_name in kb.list_frameworks():
            fw = kb.get_framework(fw_name)
            for ob in fw["obligations"]:
                assert ob["severity"] in allowed


class TestGapCheck:
    def test_gap_check_finds_missing_nda_clause(self, repo_root: Path):
        """test_gap_check_finds_missing_nda_clause finds governing_law or return_of_materials when missing."""
        kb = KnowledgeBase(knowledge_dir=str(repo_root / "knowledge"))
        missing = kb.check_gaps(["confidentiality", "term"], "nda")
        clause_types = {m.get("clause_type") for m in missing}
        assert "governing_law" in clause_types or "return_of_materials" in clause_types

    def test_gap_check_no_gaps_when_complete(self, repo_root: Path):
        """test_gap_check_no_gaps_when_complete returns [] if all required NDA clauses are present."""
        kb = KnowledgeBase(knowledge_dir=str(repo_root / "knowledge"))
        req = kb.get_required_clauses("nda")
        found = [c["clause_type"] for c in req]
        missing = kb.check_gaps(found, "nda")
        assert missing == []

    def test_gap_check_returns_severity(self, repo_root: Path):
        """test_gap_check_returns_severity ensures severity_if_missing is valid for every gap result."""
        kb = KnowledgeBase(knowledge_dir=str(repo_root / "knowledge"))
        missing = kb.check_gaps(["confidentiality", "term"], "nda")
        allowed = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        for m in missing:
            assert "severity_if_missing" in m
            assert m["severity_if_missing"] in allowed

    def test_gap_check_unknown_contract_defaults_to_nda(self, repo_root: Path):
        """test_gap_check_unknown_contract_defaults_to_nda uses default NDA template for unknown contract type."""
        kb = KnowledgeBase(knowledge_dir=str(repo_root / "knowledge"))
        missing = kb.check_gaps([], "unknown")
        assert isinstance(missing, list)
        assert len(missing) > 0

