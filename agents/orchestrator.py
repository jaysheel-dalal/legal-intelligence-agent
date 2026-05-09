import os
import json
import time
import threading
from datetime import datetime

from agents.parser_agent import ParserAgent
from agents.analysis_agent import AnalysisAgent
from agents.gap_agent import GapAgent
from agents.report_agent import ReportAgent


class Orchestrator:
    def __init__(self, llm_client, log_dir: str = "logs", output_format: str = "text"):
        self.llm_client = llm_client
        self.log_dir = log_dir
        self.output_format = output_format
        self.trace_id = (datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.agent_statuses = {
            "parser_agent": "idle",
            "analysis_agent": "idle",
            "gap_agent": "idle",
            "report_agent": "idle",
        }
        # Kill switch — set to True to
        # stop orchestration mid-flight
        self.kill_switch = False

    def run(self, query: str, file_path: str = None) -> dict:
        start = time.time()

        print()
        print("=" * 60)
        print("LEGAL INTELLIGENCE AGENT v1.0.0")
        print("PROJECT 3 — MULTI-AGENT MODE")
        print("=" * 60)
        print(f"Trace ID: {self.trace_id}")
        print()

        # ── KILL SWITCH CHECK ──────────────
        if self.kill_switch:
            print("Kill switch active. Stopping.")
            return {"error": "kill switch active"}

        # Gap queries should never be
        # rejected as vague — replace them
        # with a valid pipeline query
        original_query = None
        gap_signals = [
            "missing", "gaps", "lacking",
            "absent", "incomplete",
            "what clauses", "what clause",
            "what else", "should have",
            "what am i missing",
            "whats missing",
            "what is missing",
            "not included",
            "what need",
        ]
        query_lower = query.lower()
        if any(s in query_lower
               for s in gap_signals):
            original_query = query
            query = (
                "What are the key terms "
                "obligations and rights "
                "in this agreement?")

        # ── STEP 1: CLASSIFY QUERY ─────────
        print("[1/4] Classifying query...")
        classification = self._classify_query(query)

        if classification.get("clarification_needed"):
            print("Query too vague. Try being more specific.")
            for topic in classification.get("suggested_topics", []):
                print(f"  - {topic}")
            return {}

        intent = classification.get("intent", "liability")
        confidence = classification.get("confidence", 0.85)
        print(f"  Intent: {intent} · Confidence: {confidence}")

        # ── KILL SWITCH CHECK ──────────────
        if self.kill_switch:
            print("Kill switch active. Stopping.")
            return {"error": "kill switch active"}

        # ── STEP 2: PARSER AGENT ───────────
        print("[2/4] Running Parser Agent...")
        self.agent_statuses["parser_agent"] = "running"

        parser = ParserAgent(
            llm_client=self.llm_client,
            trace_id=self.trace_id,
            log_dir=os.path.join(self.log_dir, "traces"),
        )

        parser_result = parser.run(file_path=file_path, query=query)

        self.agent_statuses["parser_agent"] = parser_result.get("status", "failed")

        chunks = parser_result.get(
            "chunks", [])
        contract_type = parser_result.get(
            "contract_type", "unknown")
        jurisdiction = parser_result.get(
            "jurisdiction", "US — General")
        effective_date = parser_result.get(
            "effective_date", "Not found")
        expiry_date = parser_result.get(
            "expiry_date", "Not found")

        if not chunks:
            chunks = [{
                "text": query,
                "clause_type": "general",
                "position": 0,
                "char_count": len(query)
            }]

        print(
            f"  Parser: "
            f"{parser_result.get('status')} · "
            f"{len(chunks)} chunks · "
            f"type: {contract_type}"
        )

        # ── KILL SWITCH CHECK ──────────────
        if self.kill_switch:
            print("Kill switch active. Stopping.")
            return {"error": "kill switch active"}

        # ── STEP 3: PARALLEL EXECUTION ─────
        print("[3/4] Running Analysis + Gap Agents...")
        self.agent_statuses["analysis_agent"] = "running"
        self.agent_statuses["gap_agent"] = "running"

        analysis_result, gap_result = (
            self._run_parallel(
                chunks,
                query,
                intent,
                contract_type,
                jurisdiction,
                effective_date,
                expiry_date))

        self.agent_statuses["analysis_agent"] = analysis_result.get("status", "failed")
        self.agent_statuses["gap_agent"] = gap_result.get("status", "failed")

        print(
            f"  Analysis: "
            f"{analysis_result.get('status')} · "
            f"{analysis_result.get('clause_count', 0)}"
            f" clauses"
        )
        print(
            f"  Gap: "
            f"{gap_result.get('status')} · "
            f"{gap_result.get('gap_count', 0)}"
            f" missing"
        )

        gap_result = self._refine_gaps(
            analysis_result,
            gap_result,
            contract_type)

        print(
            f"  Gaps after refinement: "
            f"{gap_result.get('gap_count', 0)}")

        # ── KILL SWITCH CHECK ──────────────
        if self.kill_switch:
            print("Kill switch active. Stopping.")
            return {"error": "kill switch active"}

        # ── STEP 4: REPORT AGENT ───────────
        print("[4/4] Running Report Agent...")
        self.agent_statuses["report_agent"] = "running"

        report = ReportAgent(
            llm_client=self.llm_client,
            trace_id=self.trace_id,
            log_dir=os.path.join(self.log_dir, "traces"),
        )

        report_result = report.run(
            analysis_result=analysis_result,
            gap_result=gap_result,
            query=query,
            intent=intent,
            contract_type=contract_type,
            output_format=self.output_format,
        )

        self.agent_statuses["report_agent"] = report_result.get("status", "failed")

        # ── FINAL RESULT ───────────────────
        total_latency_ms = int((time.time() - start) * 1000)

        # Get total tokens and cost
        total_tokens = getattr(self.llm_client, "total_tokens", 0)
        total_cost = getattr(self.llm_client, "total_cost", 0.0)

        final_result = {
            "trace_id": self.trace_id,
            "query": query,
            "file_path": file_path or "none",
            "intent": intent,
            "confidence": confidence,
            "contract_type": contract_type,
            "jurisdiction": jurisdiction,
            "effective_date": effective_date,
            "expiry_date": expiry_date,
            "output": report_result.get("output", ""),
            "agent_result": report_result.get("agent_result", {}),
            "total_clauses_found": report_result.get("total_clauses_found", 0),
            "total_missing": report_result.get("total_missing", 0),
            "critical_count": report_result.get("critical_count", 0),
            "high_count": report_result.get("high_count", 0),
            "agent_statuses": self.agent_statuses,
            "total_latency_ms": total_latency_ms,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "prompt_version": "v1.0.0",
            "mode": "multi_agent_parallel",
        }

        # Restore original query in result
        if original_query is not None:
            final_result["original_query"] = (
                original_query)
            final_result["gap_analysis_mode"] = True

        # Print summary
        print()
        print("=" * 60)
        print(
            f"Clauses found: "
            f"{final_result['total_clauses_found']}"
            f"  ·  Missing: "
            f"{final_result['total_missing']}"
        )
        print(f"Latency: {total_latency_ms}ms  ·  Cost: ${total_cost:.6f}")
        print(
            f"Agents: "
            + "  ".join([
                f"{k.split('_')[0]}: {v}"
                for k, v in self.agent_statuses.items()
            ])
        )
        print("=" * 60)

        # Save trace log
        log_path = self._log_trace(final_result)
        print(f"Trace saved: {log_path}")

        return final_result

    def _classify_query(self, query: str) -> dict:
        _pipeline_gap = (
            "what are the key terms "
            "obligations and rights "
            "in this agreement?"
        )
        if query.strip().lower() == _pipeline_gap:
            return {
                "clarification_needed": False,
                "intent": "liability",
                "confidence": 0.85,
                "contract_type": "unknown",
                "entities": {
                    "time_references": [],
                    "monetary_amounts": [],
                    "party_references": [],
                },
                "suggested_topics": [],
                "raw_query": query,
            }
        try:
            return self.llm_client.classify(query)
        except Exception:
            # Better vague query handling
            query_lower = query.lower()

            # Gap queries → use liability intent
            # to avoid vague detection
            if any(w in query_lower for w in [
                "missing", "gaps", "lacking",
                "incomplete", "should have",
                "what else", "what clauses",
                "what am i missing",
            ]):
                return {
                    "clarification_needed": False,
                    "intent": "liability",
                    "confidence": 0.75,
                    "contract_type": "unknown",
                    "entities": {
                        "time_references": [],
                        "monetary_amounts": [],
                        "party_references": []
                    },
                    "suggested_topics": [],
                    "raw_query": query
                }
            return {
                "clarification_needed": False,
                "intent": "liability",
                "confidence": 0.7,
                "contract_type": "unknown",
                "entities": {
                    "time_references": [],
                    "monetary_amounts": [],
                    "party_references": [],
                },
                "suggested_topics": [],
                "raw_query": query,
            }

    def _run_parallel(self,
                      chunks: list,
                      query: str,
                      intent: str,
                      contract_type: str,
                      jurisdiction: str,
                      effective_date: str,
                      expiry_date: str
                      ) -> tuple:
        # Results containers
        analysis_result = {}
        gap_result = {}

        # Thread functions
        def run_analysis():
            agent = AnalysisAgent(
                llm_client=self.llm_client,
                trace_id=self.trace_id,
                log_dir=os.path.join(self.log_dir, "traces"),
            )
            nonlocal analysis_result
            analysis_result = agent.run(
                chunks=chunks,
                query=query,
                intent=intent,
                contract_type=contract_type,
                jurisdiction=jurisdiction,
                effective_date=effective_date,
                expiry_date=expiry_date,
            )

        def run_gap():
            agent = GapAgent(
                llm_client=self.llm_client,
                trace_id=self.trace_id,
                log_dir=os.path.join(self.log_dir, "traces"),
            )
            nonlocal gap_result
            gap_result = agent.run(
                chunks=chunks,
                contract_type=contract_type)

        # Create threads
        t_analysis = threading.Thread(target=run_analysis, name="AnalysisAgent")
        t_gap = threading.Thread(target=run_gap, name="GapAgent")

        # Start both at same time
        t_analysis.start()
        t_gap.start()

        # Wait for both to finish
        t_analysis.join(timeout=120)
        t_gap.join(timeout=120)

        # Handle timeout
        if t_analysis.is_alive():
            print("  [Orchestrator] Analysis Agent timed out")
            analysis_result = {
                "status": "failed",
                "analyzed_clauses": [],
                "clause_count": 0,
                "critical_count": 0,
                "high_count": 0,
                "error": "timeout after 120s",
            }

        if t_gap.is_alive():
            print("  [Orchestrator] Gap Agent timed out")
            gap_result = {
                "status": "failed",
                "missing_clauses": [],
                "gap_count": 0,
                "critical_gaps": 0,
                "error": "timeout after 120s",
            }

        return analysis_result, gap_result

    def _refine_gaps(
            self,
            analysis_result: dict,
            gap_result: dict,
            contract_type: str) -> dict:

        analyzed_clauses = analysis_result.get(
            "analyzed_clauses", [])

        if not analyzed_clauses:
            return gap_result

        # Get clause types that were found
        found_types = list(set([
            c.get("clause_type", "")
            for c in analyzed_clauses
            if c.get("clause_type")
        ]))

        # Filter out gaps where the
        # clause type was actually found
        current_gaps = gap_result.get(
            "missing_clauses", [])

        refined_gaps = []
        for gap in current_gaps:
            gap_type = gap.get(
                "clause_type", "").lower()
            # Check if this gap type
            # was actually found
            already_found = any(
                gap_type in found.lower()
                or found.lower() in gap_type
                for found in found_types
            )
            if not already_found:
                refined_gaps.append(gap)

        if len(refined_gaps) < len(current_gaps):
            removed = (len(current_gaps)
                       - len(refined_gaps))
            print(
                f"  [Orchestrator] "
                f"Gap refinement: removed "
                f"{removed} false positives")

        # Return updated gap result
        refined = dict(gap_result)
        refined["missing_clauses"] = (
            refined_gaps)
        refined["gap_count"] = (
            len(refined_gaps))
        refined["critical_gaps"] = sum(
            1 for g in refined_gaps
            if str(g.get(
                "severity", "")).upper()
            == "CRITICAL")
        return refined

    def _log_trace(self, result: dict) -> str:
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f"trace_{self.trace_id}.json")
        # Save without full output text
        log_data = {k: v for k, v in result.items() if k not in ["output", "agent_result"]}
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        return log_path
