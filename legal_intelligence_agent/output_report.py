from __future__ import annotations

from typing import Any, Dict, List


class ReportError(Exception):
    pass


DISCLAIMER = (
    "This is legal information, not legal advice. "
    "Consult a qualified lawyer for decisions."
)


RISK_PRIORITY = {
    "CRITICAL": 1,
    "HIGH": 2,
    "MEDIUM": 3,
    "LOW": 4,
}


class OutputReport:
    def __init__(self):
        pass

    def generate(self, agent_result: dict, format: str = "dict") -> Any:
        if agent_result is None:
            raise ReportError("agent_result cannot be None")

        if format not in {"dict", "text", "markdown"}:
            raise ReportError("invalid format")

        required_keys = {
            "query",
            "intent",
            "contract_type",
            "analyzed_clauses",
            "missing_clauses",
            "summary",
            "total_clauses_found",
            "total_missing",
            "critical_count",
            "high_count",
        }
        missing = sorted(required_keys - set(agent_result.keys()))
        if missing:
            raise ReportError(f"agent_result missing required keys: {missing}")

        if format == "dict":
            return self.to_dict(agent_result)
        if format == "text":
            return self.to_text(agent_result)
        return self.to_markdown(agent_result)

    def to_dict(self, agent_result: dict) -> Dict[str, Any]:
        clauses_in: List[dict] = list(agent_result.get("analyzed_clauses") or [])
        missing_in: List[dict] = list(agent_result.get("missing_clauses") or [])

        indexed_clauses = list(enumerate(clauses_in))
        indexed_clauses.sort(
            key=lambda t: (
                RISK_PRIORITY.get(t[1].get("risk_level"), 999),
                t[0],
            )
        )

        clauses: List[Dict[str, Any]] = []
        for _, c in indexed_clauses:
            risk_level = c.get("risk_level")
            position = c.get("position", c.get("source_position"))
            clauses.append(
                {
                    "risk_level": risk_level,
                    "clause_type": c.get("clause_type"),
                    "plain_english": c.get("plain_english", ""),
                    "source_position": position,
                    "triggered": bool(c.get("triggered", False)),
                    "confidence": float(c.get("confidence", 0.0)),
                }
            )

        indexed_missing = list(enumerate(missing_in))
        indexed_missing.sort(
            key=lambda t: (
                RISK_PRIORITY.get(t[1].get("severity"), 999),
                t[0],
            )
        )
        missing_clauses: List[Dict[str, Any]] = []
        for _, m in indexed_missing:
            missing_clauses.append(
                {
                    "severity": m.get("severity"),
                    "clause_type": m.get("clause_type"),
                    "description": m.get("description", ""),
                    "source_position": None,
                    "triggered": False,
                }
            )

        total_clauses_found = len(clauses)
        total_missing = len(missing_clauses)
        critical_count = sum(
            1
            for c in clauses
            if RISK_PRIORITY.get(c.get("risk_level")) == 1
        )
        high_count = sum(1 for c in clauses if RISK_PRIORITY.get(c.get("risk_level")) == 2)

        metadata = {
            "query": agent_result.get("query"),
            "intent": agent_result.get("intent"),
            "contract_type": agent_result.get("contract_type"),
            "total_clauses_found": total_clauses_found,
            "total_missing": total_missing,
            "critical_count": critical_count,
            "high_count": high_count,
        }

        return {
            "metadata": metadata,
            "clauses": clauses,
            "missing_clauses": missing_clauses,
            "summary": agent_result.get("summary", ""),
            "disclaimer": DISCLAIMER,
        }

    def to_text(self, agent_result: dict) -> str:
        data = self.to_dict(agent_result)

        meta = data["metadata"]
        clauses: List[Dict[str, Any]] = data["clauses"]
        missing: List[Dict[str, Any]] = data["missing_clauses"]
        summary: str = data["summary"]

        header_sep = "=" * 42
        lines: List[str] = []
        lines.append("LEGAL INTELLIGENCE AGENT — CLAUSE REPORT")
        lines.append(header_sep)
        lines.append(f"QUERY: {meta.get('query')}")
        lines.append(f"INTENT: {meta.get('intent')}")
        lines.append(f"CONTRACT TYPE: {meta.get('contract_type')}")
        lines.append(header_sep)
        lines.append("SUMMARY")
        lines.append(header_sep)
        lines.append(summary)
        lines.append(header_sep)

        # Priority order: highest to lowest.
        priority_to_header = {
            1: "CRITICAL CLAUSES",
            2: "HIGH RISK CLAUSES",
            3: "MEDIUM RISK CLAUSES",
            4: "LOW RISK CLAUSES",
        }
        for priority in [1, 2, 3, 4]:
            risk_clauses = [
                c
                for c in clauses
                if RISK_PRIORITY.get(c.get("risk_level"), 999) == priority
            ]
            if not risk_clauses:
                continue

            lines.append(priority_to_header[priority])
            for c in risk_clauses:
                position = c.get("source_position")
                lines.append(f"[{c.get('risk_level')}] {c.get('clause_type')}")
                lines.append(f"Plain English: {c.get('plain_english')}")
                lines.append(f"Source: position {position}")
                lines.append(f"Triggered: {'Yes' if c.get('triggered') else 'No'}")
                lines.append("")

        lines.append(header_sep)
        lines.append("MISSING CLAUSES")

        if not missing:
            lines.append("None found.")
        else:
            for m in missing:
                lines.append(f"[{m.get('severity')}] {m.get('clause_type')}")
                lines.append(f"Plain English: {m.get('description')}")
                lines.append("Source: position N/A")
                lines.append("Triggered: No")
                lines.append("")

        lines.append(header_sep)
        lines.append("DISCLAIMER")
        lines.append(data["disclaimer"])

        return "\n".join(lines).rstrip()

    def to_markdown(self, agent_result: dict) -> str:
        data = self.to_dict(agent_result)

        meta = data["metadata"]
        clauses: List[Dict[str, Any]] = data["clauses"]
        missing: List[Dict[str, Any]] = data["missing_clauses"]
        summary: str = data["summary"]
        disclaimer: str = data["disclaimer"]

        priority_to_emoji = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢"}
        priority_to_title = {
            1: "Critical Clauses",
            2: "High Risk Clauses",
            3: "Medium Risk Clauses",
            4: "Low Risk Clauses",
        }

        lines: List[str] = []
        lines.append("## LEGAL INTELLIGENCE AGENT — CLAUSE REPORT")
        lines.append(f"- Query: {meta.get('query')}")
        lines.append(f"- Intent: {meta.get('intent')}")
        lines.append(f"- Contract type: {meta.get('contract_type')}")
        lines.append("")
        lines.append("## Summary")
        lines.append(summary)
        lines.append("")

        for priority in [1, 2, 3, 4]:
            risk_clauses = [
                c
                for c in clauses
                if RISK_PRIORITY.get(c.get("risk_level"), 999) == priority
            ]
            if not risk_clauses:
                continue

            lines.append(f"## {priority_to_emoji[priority]} {priority_to_title[priority]}")
            for c in risk_clauses:
                lines.append(f"### {c.get('clause_type')}")
                lines.append(f"**{c.get('plain_english')}**")
                lines.append(f"- Source: position {c.get('source_position')}")
                lines.append(f"- Triggered: {'Yes' if c.get('triggered') else 'No'}")
                lines.append(f"- Confidence: {float(c.get('confidence', 0.0)):.2f}")
                lines.append("")

        lines.append("## Missing Clauses")
        if not missing:
            lines.append("> None found.")
        else:
            for m in missing:
                lines.append(f"### {m.get('clause_type')}")
                lines.append(f"**{m.get('description')}**")
                lines.append("- Source: position N/A")
                lines.append("- Triggered: No")
                lines.append("")

        lines.append("> " + disclaimer)
        return "\n".join(lines).rstrip()

