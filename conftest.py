import os
from pathlib import Path
from typing import Any, Dict, List, Sequence
from unittest.mock import MagicMock

import hashlib
import re

import pytest
import numpy as np

from legal_intelligence_agent.document_parser import DocumentParser
from legal_intelligence_agent.agent_core import AgentCore
from legal_intelligence_agent.query_intent_classifier import QueryIntentClassifier
from legal_intelligence_agent.semantic_embedder import SemanticEmbedder


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def fixtures_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def prepare_binary_fixtures(fixtures_dir: Path) -> None:
    # Tests require:
    # - empty.pdf to be 0 bytes
    # - corrupt.docx to contain malformed/random bytes
    (fixtures_dir / "empty.pdf").write_bytes(b"")
    (fixtures_dir / "corrupt.docx").write_bytes(os.urandom(128))


@pytest.fixture
def clean_nda_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "clean_nda.txt"


@pytest.fixture
def clean_vendor_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "clean_vendor.txt"


@pytest.fixture
def empty_pdf_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "empty.pdf"


@pytest.fixture
def corrupt_docx_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "corrupt.docx"


@pytest.fixture
def mock_llm_response() -> Dict[str, Any]:
    # Realistic shape for "clause analysis" style responses (mocked; never calls a real model).
    return {
        "clause_found": True,
        "clause_type": "confidentiality",
        "risk_level": "MEDIUM",
        "triggered_obligations": [],
        "legal_gaps": [],
        "source_excerpt": "Confidential Information shall be kept strictly confidential.",
    }


@pytest.fixture
def mock_llm_client(mock_llm_response: Dict[str, Any]) -> MagicMock:
    llm = MagicMock()

    def analyze_clause_chunk(text: str) -> Dict[str, Any]:
        # Keep deterministic clause typing based on content (still "LLM mocked").
        lower = text.lower()
        clause_type = "general"
        if "confidential" in lower or "non-disclosure" in lower:
            clause_type = "confidentiality"
        elif "payment" in lower or "fees" in lower or "invoice" in lower:
            clause_type = "payment"
        elif "limitation of liability" in lower or "liability" in lower:
            clause_type = "liability"
        elif "intellectual property" in lower or "ownership" in lower or "ip" in lower:
            clause_type = "IP"
        elif "terminate" in lower or "termination" in lower:
            clause_type = "termination"

        res = dict(mock_llm_response)
        res["clause_type"] = clause_type
        return res

    def classify_query(query: str) -> Dict[str, Any]:
        q = query.strip().lower()
        vague = q in {"legal stuff", "contract"} or len(q.split()) <= 2
        if vague:
            return {
                "clarification_needed": True,
                "suggested_topics": [
                    "Confidentiality and NDA terms",
                    "Payment, fees, and liability",
                    "Termination and remedies",
                ],
            }

        # Intent classification (tests assert it's one of a fixed set).
        if "terminate" in q:
            intent = "termination"
        elif "breach" in q:
            intent = "breach"
        elif "payment" in q or "fees" in q:
            intent = "payment"
        elif "intellectual property" in q or "ip" in q or "ownership" in q:
            intent = "IP"
        elif "employment" in q or "employee" in q or "employer" in q:
            intent = "employment"
        elif "liability" in q:
            intent = "liability"
        elif "data" in q or "privacy" in q:
            intent = "data"
        else:
            intent = "liability"

        return {"clarification_needed": False, "intent": intent, "suggested_topics": []}

    llm.analyze_clause_chunk.side_effect = analyze_clause_chunk
    llm.classify_query.side_effect = classify_query
    return llm


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()

    def retrieve(query: str, chunks: Sequence[Dict[str, Any]]) -> List[int]:
        # Simple "retrieval": return indices of chunks whose text contains any keywords.
        q = query.lower()
        keywords = [w for w in q.split() if len(w) >= 4]
        hits = []
        for i, c in enumerate(chunks):
            ct = c["text"].lower()
            if any(k in ct for k in keywords):
                hits.append(i)
        return hits

    embedder.retrieve.side_effect = retrieve
    return embedder


@pytest.fixture
def document_parser(mock_llm_client: MagicMock, mock_embedder: MagicMock) -> DocumentParser:
    return DocumentParser(llm_client=mock_llm_client, embedder=mock_embedder)


@pytest.fixture
def mock_embedding_client():
    """
    Deterministic embedding client for offline tests.

    - embed(text) -> numpy array of shape (384,)
    - Uses a bag-of-words hashed projection so overlapping tokens => higher cosine similarity
    """

    stopwords = {
        "the",
        "and",
        "or",
        "a",
        "an",
        "to",
        "of",
        "in",
        "for",
        "with",
        "on",
        "by",
        "shall",
        "will",
        "may",
        "must",
        "is",
        "are",
        "be",
        "as",
        "it",
        "this",
        "that",
        "between",
        "within",
        "without",
        "any",
        "all",
        "such",
        "its",
        "their",
        "party",
        "parties",
    }
    dim = 384

    def _hash_token_to_index(token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(h, 16) % dim

    def embed(text: str) -> np.ndarray:
        tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        tokens = [t for t in tokens if len(t) >= 3 and t not in stopwords]

        vec = np.zeros((dim,), dtype="float32")
        for tok in tokens:
            vec[_hash_token_to_index(tok)] += 1.0

        # Normalize to unit length to make cosine similarity equal to dot product.
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec
        vec = vec / norm
        return vec.astype("float32")

    client = MagicMock()
    client.embed.side_effect = embed
    return client


@pytest.fixture
def semantic_embedder(mock_embedding_client) -> SemanticEmbedder:
    return SemanticEmbedder(llm_client=mock_embedding_client)


@pytest.fixture
def mock_classifier_client():
    # classify(query) returns intent based on keywords with deterministic confidence.
    client = MagicMock()

    def classify(query: str) -> Dict[str, Any]:
        q = (query or "").strip().lower()
        if q in {"legal stuff", "legal help", "contract"} or len(q.split()) <= 2:
            return {
                "clarification_needed": True,
                "intent": None,
                "confidence": 0.0,
            }

        intent = "liability"
        confidence = 0.73

        if "terminate" in q or "termination" in q or "notice period" in q:
            intent, confidence = "termination", 0.91
        elif "breach" in q or "violat" in q or "fail to perform" in q:
            intent, confidence = "breach", 0.88
        elif "payment" in q or "invoice" in q or "late fee" in q or "penalt" in q:
            intent, confidence = "payment", 0.89
        elif "privacy" in q or "data" in q or "ccpa" in q:
            intent, confidence = "data", 0.87
        elif "intellectual property" in q or "copyright" in q or "ownership" in q or "ip" in q:
            intent, confidence = "IP", 0.9
        elif "employer" in q or "employee" in q or "employment" in q or "non-solicit" in q:
            intent, confidence = "employment", 0.86
        elif "liability" in q or "indemn" in q or "damages" in q:
            intent, confidence = "liability", 0.84
        elif "confidential" in q or "nda" in q or "trade secret" in q or "disclosure" in q:
            intent, confidence = "confidentiality", 0.92

        return {
            "clarification_needed": False,
            "intent": intent,
            "confidence": confidence,
        }

    client.classify.side_effect = classify
    return client


@pytest.fixture
def query_classifier(mock_classifier_client):
    return QueryIntentClassifier(llm_client=mock_classifier_client)


@pytest.fixture
def sample_nda_chunks():
    # 5 realistic NDA clause chunks.
    chunks = [
        {
            "text": "The Receiving Party shall keep all Confidential Information strictly confidential and shall not disclose it to third parties.",
            "clause_type": "confidentiality",
            "position": 0,
            "char_count": 122,
        },
        {
            "text": "Confidential Information excludes information that is public, previously known, or independently developed.",
            "clause_type": "exclusions",
            "position": 1,
            "char_count": 106,
        },
        {
            "text": "Upon termination, the Receiving Party must return or destroy all confidential materials and certify destruction.",
            "clause_type": "return_of_materials",
            "position": 2,
            "char_count": 112,
        },
        {
            "text": "This Agreement remains in effect for two years and obligations that should survive termination continue.",
            "clause_type": "term",
            "position": 3,
            "char_count": 103,
        },
        {
            "text": "Breach of this Agreement may cause irreparable harm and permits injunctive relief.",
            "clause_type": "remedies",
            "position": 4,
            "char_count": 84,
        },
    ]
    return chunks


@pytest.fixture
def sample_vendor_chunks():
    # 5 realistic vendor agreement chunks.
    return [
        {
            "text": "Customer shall pay invoices within thirty days, and late payments accrue a fee of 1.5% per month.",
            "clause_type": "payment",
            "position": 0,
            "char_count": 100,
        },
        {
            "text": "Each party's liability is capped at fees paid in the prior three months.",
            "clause_type": "liability",
            "position": 1,
            "char_count": 75,
        },
        {
            "text": "Vendor owns intellectual property in pre-existing tools and methods.",
            "clause_type": "IP",
            "position": 2,
            "char_count": 69,
        },
        {
            "text": "Either party may terminate for material breach after thirty days notice and cure opportunity.",
            "clause_type": "termination",
            "position": 3,
            "char_count": 95,
        },
        {
            "text": "Vendor will protect customer confidential data and use it only to perform services.",
            "clause_type": "data",
            "position": 4,
            "char_count": 83,
        },
    ]


@pytest.fixture
def mock_agent_llm_client():
    # analyze / translate / gap_check / validate mocked behavior for offline tests.
    client = MagicMock()

    def analyze(chunks, query, intent):
        out = []
        for c in chunks:
            t = c["text"].lower()
            if any(k in t for k in ["terminate", "termination", "breach"]):
                risk, conf = "CRITICAL", 0.95
            elif any(k in t for k in ["liability", "payment", "invoice", "late fee"]):
                risk, conf = "HIGH", 0.88
            elif any(k in t for k in ["confidential", "nda", "trade secret"]):
                risk, conf = "MEDIUM", 0.82
            else:
                risk, conf = "LOW", 0.75

            q = query.lower()
            triggered = False
            if ("terminate" in q or "termination" in q) and ("terminate" in t or "termination" in t):
                triggered = True
            if "breach" in q and "breach" in t:
                triggered = True

            out.append(
                {
                    "risk_level": risk,
                    "legal_meaning": f"This clause addresses {c.get('clause_type', 'general')} obligations and risk allocation.",
                    "triggered": triggered,
                    "confidence": conf,
                }
            )
        return out

    def translate(clause):
        text = clause.get("text", "")
        plain = (
            f"Plain explanation: {text} The clause means both sides have specific responsibilities that can affect risk and obligations."
        )
        for jargon in ["hereinafter", "pursuant", "notwithstanding", "whereas"]:
            plain = plain.replace(jargon, "plain terms")
            plain = plain.replace(jargon.title(), "plain terms")
            plain = plain.replace(jargon.upper(), "plain terms")
        return plain

    def gap_check(analyzed, contract_type):
        clause_types = {c.get("clause_type", "") for c in analyzed}
        if contract_type == "NDA":
            required = [
                ("confidentiality", "CRITICAL", "Defines confidentiality obligations and use restrictions."),
                ("exclusions", "MEDIUM", "Defines information excluded from confidentiality."),
                ("return_of_materials", "HIGH", "Requires return or destruction of confidential materials."),
                ("term", "MEDIUM", "Defines how long obligations apply."),
                ("governing_law", "MEDIUM", "Defines governing law and dispute framework."),
            ]
        elif contract_type == "vendor":
            required = [
                ("payment", "HIGH", "Specifies fees, billing cadence, and late-payment rules."),
                ("liability", "HIGH", "Defines liability caps and exclusions."),
                ("IP", "HIGH", "Allocates ownership and licensing rights."),
                ("termination", "MEDIUM", "Defines termination rights and notice/cure periods."),
            ]
        else:
            required = []

        missing = []
        for clause_type, severity, description in required:
            if clause_type not in clause_types:
                missing.append(
                    {"clause_type": clause_type, "severity": severity, "description": description}
                )
        return missing

    def validate(analyzed, translated, gaps):
        filtered = [c for c in translated if float(c.get("confidence", 0.0)) >= 0.6 and c.get("source_position") is not None]
        summary = (
            "This contract review highlights important risks and missing protections in clear language. "
            "Review critical and high-risk clauses first."
        )
        return {"analyzed_clauses": filtered, "missing_clauses": gaps, "summary": summary}

    client.analyze.side_effect = analyze
    client.translate.side_effect = translate
    client.gap_check.side_effect = gap_check
    client.validate.side_effect = validate
    return client


@pytest.fixture
def agent_core(mock_agent_llm_client):
    return AgentCore(llm_client=mock_agent_llm_client)


@pytest.fixture
def sample_agent_result():
    return {
        "query": "What happens if I terminate early?",
        "intent": "termination",
        "contract_type": "vendor",
        "total_clauses_found": 4,
        "total_missing": 2,
        "critical_count": 1,
        "high_count": 1,
        "summary": (
            "Your vendor contract has one critical "
            "termination clause requiring 30 days notice. "
            "Two standard clauses are missing."
        ),
        "analyzed_clauses": [
            {
                "text": "Either party may terminate...",
                "clause_type": "termination",
                "position": 2,
                "risk_level": "CRITICAL",
                "legal_meaning": "Termination requires notice.",
                "triggered": True,
                "plain_english": (
                    "You must give 30 days written notice "
                    "before ending this contract."
                ),
                "confidence": 0.92,
            },
            {
                "text": "Vendor shall not be liable...",
                "clause_type": "liability",
                "position": 3,
                "risk_level": "HIGH",
                "legal_meaning": "Liability is capped.",
                "triggered": False,
                "plain_english": (
                    "The vendor is not responsible for "
                    "indirect damages."
                ),
                "confidence": 0.85,
            },
            {
                "text": "Customer shall pay fees...",
                "clause_type": "payment",
                "position": 1,
                "risk_level": "MEDIUM",
                "legal_meaning": "Payment due in 30 days.",
                "triggered": False,
                "plain_english": (
                    "You must pay invoices within 30 days."
                ),
                "confidence": 0.78,
            },
            {
                "text": "This agreement is governed by...",
                "clause_type": "governing_law",
                "position": 5,
                "risk_level": "LOW",
                "legal_meaning": "California law applies.",
                "triggered": False,
                "plain_english": (
                    "This contract follows California law."
                ),
                "confidence": 0.95,
            },
        ],
        "missing_clauses": [
            {
                "clause_type": "data_protection",
                "severity": "CRITICAL",
                "description": (
                    "No data protection clause found. "
                    "Required under CCPA."
                ),
            },
            {
                "clause_type": "dispute_resolution",
                "severity": "HIGH",
                "description": (
                    "No dispute resolution clause found."
                ),
            },
        ],
    }

