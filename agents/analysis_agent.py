import os
import json
import time
from datetime import datetime


class AnalysisAgent:
    def __init__(self, llm_client, trace_id: str = None, log_dir: str = "logs/traces"):
        self.llm_client = llm_client
        self.trace_id = trace_id or (datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.log_dir = log_dir
        self.agent_id = "analysis_agent"
        self.status = "idle"
        self.tokens_used = 0
        self.cost = 0.0

    def run(
            self, chunks: list,
            query: str,
            intent: str,
            contract_type: str,
            jurisdiction: str = "US — General",
            effective_date: str = "Not found",
            expiry_date: str = "Not found") -> dict:
        self.status = "running"
        start = time.time()

        print(f"  [Analysis Agent] Starting analyze + translate...")

        try:
            # If no chunks — return empty
            if not chunks:
                self.status = "done"
                result = self._build_result([], None, start)
                self._log(result)
                return result

            # LLM Call 1 — analyze
            # (Prompt omits clause_type; see
            # ClaudeLLMClient.analyze in main_v3.)
            try:
                analysis_results = (
                    self.llm_client.analyze(
                        chunks, query,
                        intent, jurisdiction,
                        effective_date,
                        expiry_date)
                )
            except TypeError:
                try:
                    analysis_results = (
                        self.llm_client.analyze(
                            chunks, query,
                            intent, jurisdiction)
                    )
                except TypeError:
                    analysis_results = (
                        self.llm_client.analyze(
                            chunks, query, intent)
                    )

            # Build position -> chunk lookup
            # so we can match Claude's output
            # back to the parser's labels.
            # Index-based matching is unsafe
            # because Claude may return fewer
            # results or re-order them.
            chunk_by_pos = {
                c.get("position", i): c
                for i, c in enumerate(chunks)
            }

            for i, analyzed in enumerate(
                    analysis_results):
                # Find the matching chunk by
                # position; fall back to index
                pos = analyzed.get("position")
                if pos is None and i < len(chunks):
                    pos = chunks[i].get(
                        "position", i)
                    analyzed["position"] = pos

                source_chunk = chunk_by_pos.get(
                    pos,
                    chunks[i] if i < len(chunks)
                    else {})

                # Restore parser label
                # unconditionally — never trust
                # the LLM for clause_type
                analyzed["clause_type"] = (
                    source_chunk.get(
                        "clause_type",
                        "general"))

                # Backfill text/source_position
                # from chunk if missing
                if not analyzed.get("text"):
                    analyzed["text"] = (
                        source_chunk.get(
                            "text", ""))
                if analyzed.get(
                        "source_position"
                ) is None:
                    analyzed["source_position"] = (
                        pos if pos is not None
                        else i)

            print(
                f"  [Analysis Agent] Analyzed "
                f"{len(analysis_results)} clauses")

            # LLM Call 2 — translate
            try:
                translated = (
                    self.llm_client.translate(
                        analysis_results,
                        jurisdiction,
                        effective_date,
                        expiry_date)
                )
            except TypeError:
                translated = (
                    self.llm_client.translate(
                        analysis_results)
                )

            print(f"  [Analysis Agent] Translated {len(translated)} clauses")

            # Track tokens if real LLM
            if hasattr(self.llm_client, "total_tokens"):
                self.tokens_used = getattr(self.llm_client, "total_tokens", 0)
                self.cost = getattr(self.llm_client, "total_cost", 0.0)

            self.status = "done"
            result = self._build_result(translated, None, start)
            self._log(result)
            return result

        except Exception as e:
            self.status = "failed"
            print(f"  [Analysis Agent] FAILED: {e} — returning empty clauses")
            result = self._build_result([], str(e), start)
            self._log(result)
            return result

    def _build_result(self, clauses, error, start):
        latency_ms = int((time.time() - start) * 1000)
        critical = sum(1 for c in clauses if c.get("risk_level") == "CRITICAL")
        high = sum(1 for c in clauses if c.get("risk_level") == "HIGH")
        return {
            "agent_id": self.agent_id,
            "trace_id": self.trace_id,
            "status": ("failed" if error else "done"),
            "analyzed_clauses": clauses,
            "clause_count": len(clauses),
            "critical_count": critical,
            "high_count": high,
            "error": error,
            "latency_ms": latency_ms,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
        }

    def _log(self, result: dict) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f"analysis_{self.trace_id}.json")
        # Remove large clause list from log
        log_data = {k: v for k, v in result.items() if k != "analyzed_clauses"}
        log_data["clause_count"] = result.get("clause_count", 0)
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
