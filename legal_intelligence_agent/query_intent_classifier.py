from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from legal_intelligence_agent.text_signals import any_signal, count_signals


class ClassificationError(Exception):
    pass


class QueryIntentClassifier:
    VALID_INTENTS = {
        "termination",
        "breach",
        "payment",
        "data",
        "IP",
        "employment",
        "liability",
        "confidentiality",
    }

    KNOWN_VAGUE_PHRASES = {
        "legal stuff",
        "legal help",
        "contract help",
        "need legal advice",
        "help me",
        "question",
    }

    LEGAL_KEYWORDS = {
        "terminate",
        "termination",
        "breach",
        "payment",
        "fee",
        "invoice",
        "privacy",
        "data",
        "ccpa",
        "intellectual property",
        "copyright",
        "employment",
        "employee",
        "employer",
        "liability",
        "indemn",
        "nda",
        "confidential",
        "non-solicit",
        "contract",
        "agreement",
    }

    PARTY_KEYWORDS = {
        "vendor",
        "supplier",
        "customer",
        "employee",
        "employer",
        "contractor",
        "client",
        "buyer",
        "seller",
        "licensor",
        "licensee",
        "disclosing party",
        "receiving party",
        "company",
        "party",
    }

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client

    def _detect_intent_from_query(
            self, query: str) -> str:

        q = query.lower().strip()

        # Remove common filler words
        q = q.replace("what are", "")
        q = q.replace("what is", "")
        q = q.replace("tell me", "")
        q = q.replace("show me", "")
        q = q.replace("can you", "")
        q = q.replace("please", "")
        q = q.strip()

        intent_signals = {
            "payment": [
                "pay", "paid", "salary",
                "compensation", "benefit",
                "bonus", "royalt", "fee",
                "invoice", "price", "cost",
                "money", "cash", "earn",
                "income", "wage", "stipend",
                "reimburs", "allowance",
                "subscription", "rate",
                "financial", "dollar",
            ],
            "termination": [
                "terminat", "fire", "fired",
                "end", "quit", "resign",
                "resign", "laid off",
                "layoff", "dismiss",
                "exit", "leave", "expire",
                "cancel", "notice period",
                "severance", "separation",
                "what happen", "if i leave",
                "if i quit", "if fired",
                "early terminat",
            ],
            "confidentiality": [
                "confidential", "secret",
                "nda", "disclose",
                "disclosure", "privacy",
                "proprietary", "sensitive",
                "keep secret", "not share",
                "obligation", "restrict",
            ],
            "IP": [
                "ip", "patent", "invention",
                "ownership", "own",
                "copyright", "trademark",
                "intellectual property",
                "work product", "create",
                "develop", "invent",
                "who owns", "belong",
                "license", "licens",
                "rights to",
            ],
            "liability": [
                "liab", "risk", "damage",
                "responsible", "responsib",
                "indemnif", "warrant",
                "guarantee", "protect",
                "sue", "lawsuit", "claim",
                "dispute", "legal action",
                "penalty", "fine",
                "missing", "gap", "absent",
                "incomplete", "not include",
                "what else", "what clause",
                "should have", "what need",
            ],
            "governing_law": [
                "govern", "law", "legal",
                "jurisdiction", "state law",
                "which court", "arbitrat",
                "mediat", "dispute resolut",
                "ohio", "delaware",
                "new york", "california",
            ],
            "employment": [
                "role", "job", "position",
                "duti", "responsib",
                "report to", "title",
                "work hour", "schedule",
                "full time", "part time",
                "probat", "performance",
                "review", "promot",
            ],
        }

        # Score each intent. Word-boundary matching, so "end" does not score
        # the termination intent for a question about a vendor.
        scores = {}
        for intent, signals in (
                intent_signals.items()):
            score = count_signals(q, signals)
            if score > 0:
                scores[intent] = score

        if not scores:
            return "liability"

        return max(
            scores, key=scores.get)

    def is_vague(self, query: str) -> bool:
        q = query.strip().lower()

        intent_signals = {
            "payment": [
                "pay", "paid", "salary",
                "compensation", "benefit",
                "bonus", "royalt", "fee",
                "invoice", "price", "cost",
                "money", "cash", "earn",
                "income", "wage", "stipend",
                "reimburs", "allowance",
                "subscription", "rate",
                "financial", "dollar",
            ],
            "termination": [
                "terminat", "fire", "fired",
                "end", "quit", "resign",
                "resign", "laid off",
                "layoff", "dismiss",
                "exit", "leave", "expire",
                "cancel", "notice period",
                "severance", "separation",
                "what happen", "if i leave",
                "if i quit", "if fired",
                "early terminat",
            ],
            "confidentiality": [
                "confidential", "secret",
                "nda", "disclose",
                "disclosure", "privacy",
                "proprietary", "sensitive",
                "keep secret", "not share",
                "obligation", "restrict",
            ],
            "IP": [
                "ip", "patent", "invention",
                "ownership", "own",
                "copyright", "trademark",
                "intellectual property",
                "work product", "create",
                "develop", "invent",
                "who owns", "belong",
                "license", "licens",
                "rights to",
            ],
            "liability": [
                "liab", "risk", "damage",
                "responsible", "responsib",
                "indemnif", "warrant",
                "guarantee", "protect",
                "sue", "lawsuit", "claim",
                "dispute", "legal action",
                "penalty", "fine",
                "missing", "gap", "absent",
                "incomplete", "not include",
                "what else", "what clause",
                "should have", "what need",
            ],
            "governing_law": [
                "govern", "law", "legal",
                "jurisdiction", "state law",
                "which court", "arbitrat",
                "mediat", "dispute resolut",
                "ohio", "delaware",
                "new york", "california",
            ],
            "employment": [
                "role", "job", "position",
                "duti", "responsib",
                "report to", "title",
                "work hour", "schedule",
                "full time", "part time",
                "probat", "performance",
                "review", "promot",
            ],
        }

        # Only flag as vague if truly empty
        # or gibberish — not if it has
        # recognizable legal intent words
        all_signals = [
            s for signals in
            intent_signals.values()
            for s in signals]

        has_intent = any_signal(q, all_signals)

        if "contract" in q or "agreement" in q:
            has_intent = True

        if not has_intent and len(q) < 10:
            return True

        # Otherwise always proceed
        return False

    def extract_entities(self, query: str) -> dict:
        # Regex-only extraction: no LLM calls.
        q = query or ""
        lower = q.lower()

        # Time references: "30 days", "2 years", "within 10 business days", "per month"
        time_patterns = [
            r"\b\d+\s*(?:business\s+)?(?:day|days|week|weeks|month|months|year|years)\b",
            r"\bwithin\s+\d+\s*(?:business\s+)?(?:day|days|week|weeks|month|months|year|years)\b",
            r"\bper\s+(?:day|week|month|year)\b",
        ]
        time_refs: List[str] = []
        for pat in time_patterns:
            for m in re.findall(pat, lower, flags=re.IGNORECASE):
                time_refs.append(m)

        # Monetary amounts: "$500", "$1,200.50", "1.5%", "25 %"
        money_patterns = [
            r"\$\s?\d[\d,]*(?:\.\d+)?",
            r"\b\d+(?:\.\d+)?\s?%",
        ]
        monetary: List[str] = []
        for pat in money_patterns:
            for m in re.findall(pat, q, flags=re.IGNORECASE):
                monetary.append(m.strip())

        # Party references from controlled list.
        parties: List[str] = []
        for p in self.PARTY_KEYWORDS:
            if re.search(rf"\b{re.escape(p)}\b", lower):
                parties.append(p)

        return {
            "time_references": list(dict.fromkeys(time_refs)),
            "monetary_amounts": list(dict.fromkeys(monetary)),
            "party_references": sorted(list(dict.fromkeys(parties))),
        }

    def _extract_contract_type(self, query: str) -> str:
        q = (query or "").lower()
        if "nda" in q or "non-disclosure" in q or "confidentiality agreement" in q:
            return "NDA"
        if "saas" in q or "software as a service" in q or "subscription agreement" in q:
            return "SaaS"
        if "employment" in q or "employee" in q or "employer" in q:
            return "employment"
        if "vendor" in q or "supplier" in q or "services agreement" in q:
            return "vendor"
        return "unknown"

    def _suggested_topics(self) -> List[str]:
        return [
            "Termination and notice periods",
            "Payment terms and late fees",
            "Liability, indemnification, and risk caps",
            "Confidentiality and NDA obligations",
        ]

    def classify(self, query: str) -> dict:
        if query is None:
            raise ClassificationError("query cannot be None")

        raw_query = query
        query = query.strip()
        if not query:
            raise ClassificationError("query cannot be empty")

        entities = self.extract_entities(query)
        contract_type = self._extract_contract_type(query)

        query_lower = query.lower()

        # Gap and natural human queries
        # should never be flagged as vague
        gap_signals = [
            "missing", "gaps", "lacking",
            "absent", "incomplete",
            "what clauses", "what clause",
            "what else", "should have",
            "what am i missing",
            "whats missing",
            "what is missing",
            "not included", "what need",
            "anything else", "did i miss",
            "what should", "what could",
            "what would", "what might",
            "is there anything",
            "have i covered",
            "am i covered",
            "what protection",
            "am i protected",
            "what rights",
            "what obligations",
            "what are my",
            "what do i",
            "what can i",
            "what happens if",
            "if i get",
            "if they",
            "can they",
            "can i",
            "do i have",
            "am i allowed",
            "are they allowed",
            "how long",
            "how much",
            "who owns",
            "who pays",
            "who is responsible",
        ]

        skip_vague = any(
            s in query_lower for s in gap_signals)

        if not skip_vague and self.is_vague(query):
            return {
                "clarification_needed": True,
                "intent": None,
                "confidence": 0.0,
                "contract_type": contract_type,
                "entities": entities,
                "suggested_topics": self._suggested_topics(),
                "raw_query": raw_query,
            }

        if self.llm_client is None or not hasattr(self.llm_client, "classify"):
            intent = self._detect_intent_from_query(query)
            return {
                "clarification_needed": False,
                "intent": intent,
                "confidence": 0.75,
                "contract_type": contract_type,
                "entities": entities,
                "suggested_topics": [],
                "raw_query": raw_query,
            }

        try:
            model_out = self.llm_client.classify(query)
        except Exception as e:  # noqa: BLE001
            raise ClassificationError(f"classification failed: {e}") from e

        intent = model_out.get("intent")
        confidence = float(model_out.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        if intent not in self.VALID_INTENTS:
            raise ClassificationError("invalid intent returned by llm_client")

        return {
            "clarification_needed": False,
            "intent": intent,
            "confidence": confidence,
            "contract_type": contract_type,
            "entities": entities,
            "suggested_topics": [],
            "raw_query": raw_query,
        }

