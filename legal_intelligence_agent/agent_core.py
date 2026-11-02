from __future__ import annotations

from typing import Any, Dict, List

from legal_intelligence_agent.confidence import (
    drop_counts,
    model_confidence,
    partition_by_threshold,
)


class AgentError(Exception):
    pass


class AgentCore:
    JARGON_WORDS = ["hereinafter", "pursuant", "notwithstanding", "whereas"]

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def run(
        self, query: str, intent: str, chunks: list[dict], contract_type: str = "unknown"
    ) -> dict:
        if self.llm_client is None:
            raise AgentError("llm_client is required")
        if not chunks:
            raise AgentError("chunks cannot be empty")

        analyzed = self.analyze(chunks, query, intent)
        translated = self.translate(analyzed)
        gaps = self.gap_check(analyzed, contract_type)
        validated = self.validate(analyzed, translated, gaps)

        analyzed_clauses = validated.get("analyzed_clauses", [])
        missing_clauses = validated.get("missing_clauses", [])

        critical_count = sum(1 for c in analyzed_clauses if c.get("risk_level") == "CRITICAL")
        high_count = sum(1 for c in analyzed_clauses if c.get("risk_level") == "HIGH")
        total_clauses_found = len(analyzed_clauses)
        total_missing = len(missing_clauses)
        summary = validated.get(
            "summary",
            f"This review found {total_clauses_found} relevant clauses and {total_missing} missing protections.",
        )
        for word in self.JARGON_WORDS:
            summary = summary.replace(word, " ")

        return {
            "query": query,
            "intent": intent,
            "contract_type": contract_type,
            "analyzed_clauses": analyzed_clauses,
            "missing_clauses": missing_clauses,
            "summary": " ".join(summary.split()),
            "total_clauses_found": total_clauses_found,
            "total_missing": total_missing,
            "critical_count": critical_count,
            "high_count": high_count,
        }

    def analyze(self, chunks: list[dict], query: str, intent: str) -> list[dict]:
        if not chunks:
            raise AgentError("chunks cannot be empty")
        if self.llm_client is None or not hasattr(self.llm_client, "analyze"):
            raise AgentError("llm_client.analyze is required")

        try:
            analyzed = self.llm_client.analyze(chunks, query, intent)
        except Exception as e:  # noqa: BLE001
            raise AgentError(f"analyze failed: {e}") from e

        out: List[Dict[str, Any]] = []
        for chunk, item in zip(chunks, analyzed):
            clause = dict(chunk)
            clause["risk_level"] = item.get("risk_level", "LOW")
            clause["legal_meaning"] = item.get("legal_meaning", "")
            clause["triggered"] = bool(item.get("triggered", False))
            clause["source_position"] = chunk.get("position")
            # None when the model reported nothing usable. This used to default
            # to 0.7, which silently cleared the 0.6 gate below, so an unscored
            # clause was indistinguishable from one the model vouched for.
            clause["confidence"] = model_confidence(item)
            out.append(clause)
        return out

    def translate(self, analyzed: list[dict]) -> list[dict]:
        if self.llm_client is None or not hasattr(self.llm_client, "translate"):
            raise AgentError("llm_client.translate is required")

        translated: List[Dict[str, Any]] = []
        for clause in analyzed:
            try:
                plain = self.llm_client.translate(clause)
            except Exception as e:  # noqa: BLE001
                raise AgentError(f"translate failed: {e}") from e
            if not isinstance(plain, str):
                plain = str(plain)

            clean = plain
            for word in self.JARGON_WORDS:
                clean = clean.replace(word, "plain terms")

            new_clause = dict(clause)
            new_clause["plain_english"] = " ".join(clean.split())
            translated.append(new_clause)
        return translated

    def gap_check(self, analyzed: list[dict], contract_type: str) -> list[dict]:
        if self.llm_client is None or not hasattr(self.llm_client, "gap_check"):
            raise AgentError("llm_client.gap_check is required")
        try:
            gaps = self.llm_client.gap_check(analyzed, contract_type)
        except Exception as e:  # noqa: BLE001
            raise AgentError(f"gap_check failed: {e}") from e

        out = []
        for g in gaps or []:
            out.append(
                {
                    "clause_type": g.get("clause_type", "unknown"),
                    "severity": g.get("severity", "MEDIUM"),
                    "description": g.get("description", ""),
                }
            )
        return out

    def validate(self, analyzed: list[dict], translated: list[dict], gaps: list[dict]) -> dict:
        if self.llm_client is None or not hasattr(self.llm_client, "validate"):
            raise AgentError("llm_client.validate is required")
        try:
            validated = self.llm_client.validate(analyzed, translated, gaps)
        except Exception as e:  # noqa: BLE001
            raise AgentError(f"validate failed: {e}") from e

        clauses = validated.get("analyzed_clauses", translated)

        # A clause with no model confidence fails the gate, same as one scored
        # below it. float(c.get("confidence", 0.0)) also raised on a
        # non-numeric value instead of rejecting the clause.
        scored, dropped_missing, dropped_low = partition_by_threshold(clauses)
        filtered = [c for c in scored if c.get("source_position") is not None]

        return {
            "analyzed_clauses": filtered,
            "missing_clauses": validated.get("missing_clauses", gaps),
            "summary": validated.get(
                "summary",
                "The contract review highlights key risks and missing protections in plain language.",
            ),
            **drop_counts(dropped_missing, dropped_low),
        }

