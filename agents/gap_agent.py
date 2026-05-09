import os
import json
import time
from datetime import datetime

from legal_intelligence_agent.knowledge_base import KnowledgeBase


class GapAgent:
    def __init__(self, llm_client, trace_id: str = None, log_dir: str = "logs/traces"):
        self.llm_client = llm_client
        self.trace_id = trace_id or (datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.log_dir = log_dir
        self.agent_id = "gap_agent"
        self.status = "idle"
        self.tokens_used = 0
        self.cost = 0.0

    def run(self, chunks: list, contract_type: str, analyzed_clauses: list = None) -> dict:
        self.status = "running"
        start = time.time()

        print(f"  [Gap Agent] Starting gap check for {contract_type}...")

        try:
            # If no chunks — return empty gaps
            if not chunks:
                self.status = "done"
                result = self._build_result([], None, start)
                self._log(result)
                return result

            kb = KnowledgeBase("knowledge")
            template = kb.get_required_clauses(contract_type)

            missing = []
            for required_clause in template:
                clause_name = required_clause.get(
                    "clause_type", "")

                is_present = self._is_clause_present(
                    clause_name, chunks)

                print(f"  [GAP DEBUG] {clause_name}: "
                      f"{'FOUND' if is_present else 'MISSING'}")

                if not is_present:
                    missing.append(required_clause)

            gaps = missing

            print(f"  [Gap Agent] Found {len(gaps)} missing clauses")

            # Track tokens if real LLM
            if hasattr(self.llm_client, "total_tokens"):
                self.tokens_used = getattr(self.llm_client, "total_tokens", 0)
                self.cost = getattr(self.llm_client, "total_cost", 0.0)

            self.status = "done"
            result = self._build_result(gaps, None, start)
            self._log(result)
            return result

        except Exception as e:
            self.status = "failed"
            print(f"  [Gap Agent] FAILED: {e} — returning empty gaps")
            result = self._build_result([], str(e), start)
            self._log(result)
            return result

    def _is_clause_present(
            self,
            clause_name: str,
            chunks: list) -> bool:

        presence_signals = {
            # Employment
            "compensation_structure": [
                "base salary", "annual salary",
                "per year", "payroll",
                "compensation", "wages",
                "salary of", "shall pay",
            ],
            "benefits_and_equity": [
                "health", "insurance",
                "dental", "disability",
                "retirement", "401k",
                "benefits", "stock option",
                "equity", "rsu", "benefit plan",
            ],
            "at_will_employment": [
                "at-will", "at will",
                "without cause",
                "terminate at any time",
                "employment at will",
            ],
            "non_compete": [
                "non-compete", "noncompete",
                "competitive activities",
                "not compete", "competition",
                "shall not engage",
                "restricted period",
                "direct competition",
                "competitive activit",
            ],
            "non_compete_agreement": [
                "non-compete", "noncompete",
                "competitive activities",
                "not compete", "competition",
                "shall not engage",
                "restricted period",
            ],
            "non_solicitation": [
                "non-solicitation",
                "nonsolicitation",
                "solicit", "not solicit",
                "recruit", "soliciting",
                "hire any employee",
                "encourage to leave",
            ],
            "non_solicitation_agreement": [
                "non-solicitation",
                "nonsolicitation",
                "solicit", "not solicit",
                "recruit", "soliciting",
                "hire any employee",
                "encourage to leave",
            ],
            "restrictive_covenants": [
                "restrictive covenant",
                "non-compete", "non-solicitation",
                "solicit", "competitive",
                "restricted period",
            ],
            "severance_and_separation": [
                "severance", "separation pay",
                "continuation of salary",
                "termination benefits",
                "upon termination",
                "twelve months",
                "12 months",
            ],
            "dispute_resolution": [
                "arbitration", "mediation",
                "dispute resolution",
                "aaa", "jams",
                "binding arbitration",
                "american arbitration",
            ],
            "intellectual_property_ownership": [
                "intellectual property",
                "work product", "inventions",
                "ip ownership", "assign",
                "belongs to company",
                "owned by employer",
                "all inventions",
            ],
            "work_duties_and_role": [
                "duties", "responsibilities",
                "job title", "position",
                "reporting to", "shall perform",
                "executive vice president",
                "devote", "services",
            ],
            "governing_law": [
                "governing law", "governed by",
                "laws of the state",
                "jurisdiction", "venue",
                "choice of law",
                "applicable law",
                "state of",
                "commonwealth of",
            ],
            "confidentiality": [
                "confidential", "keep secret",
                "not disclose", "trade secret",
                "confidential information",
                "strictest confidence",
                "proprietary",
            ],
            "confidential_information": [
                "confidential information",
                "proprietary information",
                "trade secret", "confidential",
            ],
            "termination": [
                "terminat", "fired",
                "without cause", "for cause",
                "notice of termination",
                "termination of employment",
            ],
            "change_in_control": [
                "change in control",
                "change of control",
                "acquisition", "merger",
                "lump sum", "buyout",
            ],
            # IP License
            "license_grant": [
                "hereby grants", "grant of",
                "license to", "licensor grants",
                "exclusive license",
                "non-exclusive license",
                "grants to",
            ],
            "royalty_payment": [
                "royalty", "royalties",
                "license fee", "milestone",
                "milestone payment",
            ],
            "audit_rights": [
                "audit", "inspect",
                "right to audit",
                "audit records",
                "once per year",
            ],
            "indemnification": [
                "indemnif", "hold harmless",
                "defend", "indemnitor",
                "save and defend",
                "losses and damages",
                "third party claim",
                "indemnify and defend",
                "claims arising",
                "liabilities arising",
            ],
            "warranty_disclaimer": [
                "warrant", "disclaimer",
                "as is", "no warranty",
                "represents and warrants",
                "warranties",
            ],
            # NDA
            "return_of_materials": [
                "return", "destroy",
                "return all",
                "certify destruction",
                "return or destroy",
            ],
            "permitted_disclosure": [
                "permitted disclosure",
                "required by law",
                "court order",
                "legal process",
                "required to disclose",
            ],
            # Vendor
            "liability": [
                "liability", "liable",
                "limitation of liability",
                "in no event", "shall not exceed",
                "aggregate liability",
                "indirect damages",
                "consequential",
            ],
            "limitation_of_liability": [
                "limitation of liability",
                "limit liability",
                "liability cap",
                "in no event",
                "shall not exceed",
                "aggregate liability",
                "indirect damages",
                "consequential damages",
                "special damages",
            ],
            "force_majeure": [
                "force majeure",
                "act of god",
                "beyond reasonable control",
                "unforeseeable",
                "natural disaster",
                "pandemic", "epidemic",
                "government action",
                "circumstances beyond",
                "outside reasonable control",
                "beyond the control",
                "not reasonably foreseeable",
            ],
            "warranties": [
                "warrant", "warranty",
                "represents and warrants",
                "fit for purpose",
                "merchantability",
                "as is", "no warranty",
                "limited warranty",
                "warranty disclaimer",
            ],
            # SaaS
            "uptime_sla": [
                "uptime", "availability",
                "service level",
                "sla", "99.",
                "uptime percentage",
            ],
            "data_security": [
                "data security", "encryption",
                "data protection",
                "security measures",
                "cyber", "breach",
            ],
        }

        signals = presence_signals.get(
            clause_name, [])

        if not signals:
            signals = clause_name.replace(
                "_", " ").split()

        for chunk in chunks:
            text = chunk.get(
                "text", "").lower()
            if any(
                signal.lower() in text
                for signal in signals
            ):
                return True

        return False

    def _build_result(self, gaps, error, start):
        latency_ms = int((time.time() - start) * 1000)
        critical_gaps = sum(
            1 for g in gaps
            if str(
                g.get("severity")
                or g.get("severity_if_missing", "")
            ).upper() == "CRITICAL"
        )
        return {
            "agent_id": self.agent_id,
            "trace_id": self.trace_id,
            "status": ("failed" if error else "done"),
            "missing_clauses": gaps,
            "gap_count": len(gaps),
            "critical_gaps": critical_gaps,
            "error": error,
            "latency_ms": latency_ms,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
        }

    def _log(self, result: dict) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f"gap_{self.trace_id}.json")
        log_data = {k: v for k, v in result.items() if k != "missing_clauses"}
        log_data["gap_count"] = result.get("gap_count", 0)
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
