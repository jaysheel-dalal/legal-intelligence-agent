import os
import json
from datetime import datetime


class ApprovalGate:
    def __init__(self, confidence_threshold: float = 0.92, log_dir: str = "logs"):
        self.confidence_threshold = confidence_threshold
        self.log_dir = log_dir
        self.decisions = []

    def review(self, agent_result: dict) -> dict:
        if not agent_result:
            raise ValueError("agent_result is None")

        clauses = agent_result.get("analyzed_clauses", [])
        approved_clauses = []
        rejected_clauses = []

        print()
        print("=" * 60)
        print("APPROVAL GATE — HUMAN REVIEW")
        print("=" * 60)

        critical_high = [
            c for c in clauses
            if c.get("risk_level") in {"CRITICAL", "HIGH"}
            and not self._auto_approve(c)
        ]
        auto_approved = [c for c in clauses if self._auto_approve(c)]

        print(f"Total clauses: {len(clauses)}")
        print(f"Auto-approved: {len(auto_approved)} (LOW/MEDIUM risk)")
        print(f"Needs review:  {len(critical_high)} (CRITICAL/HIGH risk)")
        print()

        # Auto-approve LOW and MEDIUM
        for clause in auto_approved:
            self._log_decision(
                clause, "APPROVED",
                "Auto-approved: LOW/MEDIUM risk",
                auto=True)
            approved_clauses.append(clause)

        # Human review for CRITICAL and HIGH
        if critical_high:
            print("Review each clause below.")
            print("Commands: APPROVE / REJECT / OVERRIDE <reason>")
            print("-" * 60)

            for i, clause in enumerate(critical_high, 1):
                decision_input = self._present_clause(clause, i, len(critical_high))
                decision_input = decision_input.strip().upper()

                if decision_input.startswith("OVERRIDE"):
                    reason = decision_input[8:].strip()
                    if not reason:
                        reason = "Human override"
                    self._log_decision(clause, "OVERRIDE", reason, auto=False)
                    # Override — include with note
                    clause["human_override"] = True
                    clause["override_reason"] = reason
                    approved_clauses.append(clause)
                    print(f"  ↳ OVERRIDE recorded: {reason}")

                elif decision_input == "REJECT":
                    reason = input("  Rejection reason: ").strip()
                    if not reason:
                        reason = "Rejected by reviewer"
                    self._log_decision(clause, "REJECTED", reason, auto=False)
                    rejected_clauses.append(clause)
                    print(f"  ↳ REJECTED: {reason}")

                else:
                    # Default to APPROVE
                    self._log_decision(clause, "APPROVED", "Human approved", auto=False)
                    approved_clauses.append(clause)
                    print("  ↳ APPROVED")
        else:
            print("No CRITICAL/HIGH clauses require human review.")

        # Save log
        log_file = self._save_log()

        # Build final result
        final_result = dict(agent_result)
        final_result["analyzed_clauses"] = approved_clauses
        final_result["rejected_clauses"] = rejected_clauses
        final_result["total_clauses_found"] = len(approved_clauses)
        final_result["critical_count"] = sum(
            1 for c in approved_clauses if c.get("risk_level") == "CRITICAL"
        )
        final_result["high_count"] = sum(
            1 for c in approved_clauses if c.get("risk_level") == "HIGH"
        )
        final_result["approval_log"] = log_file
        final_result["approval_stats"] = {
            "total": len(clauses),
            "auto_approved": len(auto_approved),
            "human_approved": sum(
                1 for d in self.decisions if d["decision"] == "APPROVED" and not d["auto"]
            ),
            "overrides": sum(1 for d in self.decisions if d["decision"] == "OVERRIDE"),
            "rejected": len(rejected_clauses),
        }

        print()
        print("=" * 60)
        print("APPROVAL SUMMARY")
        print("=" * 60)
        print(f"Approved:  {len(approved_clauses)}")
        print(f"Rejected:  {len(rejected_clauses)}")
        print(f"Overrides: {final_result['approval_stats']['overrides']}")
        print(f"Log saved: {log_file}")
        print("=" * 60)

        return final_result

    def _auto_approve(self, clause: dict) -> bool:
        risk = clause.get("risk_level", "LOW")
        confidence = clause.get("confidence", 0)

        # Always auto-approve LOW and MEDIUM
        if risk in {"LOW", "MEDIUM"}:
            return True

        # Auto-approve HIGH if very high confidence
        if risk == "HIGH" and confidence >= self.confidence_threshold:
            return True

        # CRITICAL always needs human review
        return False

    def _present_clause(self, clause: dict, index: int, total: int) -> str:
        risk = clause.get("risk_level", "UNKNOWN")
        risk_icons = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢",
        }
        icon = risk_icons.get(risk, "⚪")

        print()
        print(f"[{index}/{total}] {icon} {risk} — {clause.get('clause_type', 'unknown')}")
        print(f"Plain English: {clause.get('plain_english', 'N/A')}")
        print(f"Source:        position {clause.get('source_position', 'N/A')}")
        print(f"Confidence:    {clause.get('confidence', 0):.2f}")
        print()

        return input("Decision (APPROVE/REJECT/OVERRIDE <reason>): ")

    def _log_decision(self, clause: dict, decision: str, reason: str, auto: bool) -> None:
        self.decisions.append({
            "timestamp": datetime.now().isoformat(),
            "clause_type": clause.get("clause_type", "unknown"),
            "risk_level": clause.get("risk_level", "unknown"),
            "confidence": clause.get("confidence", 0),
            "decision": decision,
            "reason": reason,
            "auto": auto,
            "source_position": clause.get("source_position", 0),
        })

    def _save_log(self) -> str:
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"approvals_{timestamp}.json")

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "total_decisions": len(self.decisions),
            "approved": sum(1 for d in self.decisions if d["decision"] == "APPROVED"),
            "rejected": sum(1 for d in self.decisions if d["decision"] == "REJECTED"),
            "overrides": sum(1 for d in self.decisions if d["decision"] == "OVERRIDE"),
            "auto_approved": sum(1 for d in self.decisions if d["auto"]),
            "decisions": self.decisions,
        }

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

        return log_file
