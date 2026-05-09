import os
import json
import time
from datetime import datetime


class ReportAgent:
    def __init__(self, llm_client, trace_id: str = None, log_dir: str = "logs/traces"):
        self.llm_client = llm_client
        self.trace_id = trace_id or (datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.log_dir = log_dir
        self.agent_id = "report_agent"
        self.status = "idle"
        self.tokens_used = 0
        self.cost = 0.0

    def run(
        self,
        analysis_result: dict,
        gap_result: dict,
        query: str,
        intent: str,
        contract_type: str,
        output_format: str = "text",
    ) -> dict:
        self.status = "running"
        start = time.time()

        print(f"  [Report Agent] Starting validate + report...")

        try:
            # Get analyzed clauses from AnalysisAgent — use empty list
            # if analysis failed (circuit breaker)
            analyzed = analysis_result.get("analyzed_clauses", [])

            # Get gaps from GapAgent —
            # use empty list if gap failed
            gaps = gap_result.get("missing_clauses", [])

            print(f"  [Report Agent] Validating {len(analyzed)} clauses + {len(gaps)} gaps")

            # LLM Call — validate
            validated = self.llm_client.validate(analyzed, analyzed, gaps)

            # Add required fields
            validated["query"] = query
            validated["intent"] = intent
            validated["contract_type"] = contract_type

            # Note which agents failed
            failed_agents = []
            if analysis_result.get("status") == "failed":
                failed_agents.append("analysis_agent")
            if gap_result.get("status") == "failed":
                failed_agents.append("gap_agent")
            if failed_agents:
                validated["degraded_mode"] = True
                validated["failed_agents"] = failed_agents
                print(
                    f"  [Report Agent] Warning: running in degraded mode — {failed_agents} failed"
                )

            # Generate formatted output
            from legal_intelligence_agent.output_report import OutputReport

            reporter = OutputReport()
            output = reporter.generate(validated, format=output_format)

            # Track tokens if real LLM
            if hasattr(self.llm_client, "total_tokens"):
                self.tokens_used = getattr(self.llm_client, "total_tokens", 0)
                self.cost = getattr(self.llm_client, "total_cost", 0.0)

            self.status = "done"
            latency_ms = int((time.time() - start) * 1000)

            result = {
                "agent_id": self.agent_id,
                "trace_id": self.trace_id,
                "status": "done",
                "output": output,
                "agent_result": validated,
                "total_clauses_found": validated.get("total_clauses_found", 0),
                "total_missing": validated.get("total_missing", 0),
                "critical_count": validated.get("critical_count", 0),
                "high_count": validated.get("high_count", 0),
                "error": None,
                "latency_ms": latency_ms,
                "tokens_used": self.tokens_used,
                "cost": self.cost,
            }

            self._log(result)
            return result

        except Exception as e:
            self.status = "failed"
            latency_ms = int((time.time() - start) * 1000)
            print(f"  [Report Agent] FAILED: {e}")
            result = {
                "agent_id": self.agent_id,
                "trace_id": self.trace_id,
                "status": "failed",
                "output": "Report generation failed. Please try again.",
                "agent_result": {},
                "total_clauses_found": 0,
                "total_missing": 0,
                "critical_count": 0,
                "high_count": 0,
                "error": str(e),
                "latency_ms": latency_ms,
                "tokens_used": self.tokens_used,
                "cost": self.cost,
            }
            self._log(result)
            return result

    def _log(self, result: dict) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f"report_{self.trace_id}.json")
        # Don't log full output text
        log_data = {k: v for k, v in result.items() if k not in ["output", "agent_result"]}
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
