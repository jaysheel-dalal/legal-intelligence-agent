import os
import re
import json
import time
import hashlib
from datetime import datetime


class ParserAgent:
    def __init__(self, llm_client, trace_id: str = None, log_dir: str = "logs/traces"):
        self.llm_client = llm_client
        self.trace_id = trace_id or (datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.log_dir = log_dir
        self.agent_id = "parser_agent"
        self.status = "idle"
        # idle / running / done / failed

    def run(self, file_path: str = None, query: str = "") -> dict:
        self.status = "running"
        start = time.time()

        print(f"  [Parser Agent] Starting...")

        try:
            from legal_intelligence_agent.document_parser import DocumentParser

            parser = DocumentParser(llm_client=self.llm_client)

            if file_path:
                doc = parser.parse_document(document_path=file_path)
                chunks = doc.get("chunks", [])
                contract_type = doc.get("contract_type", "unknown")
                jurisdiction = doc.get(
                    "jurisdiction",
                    "US — General")
                effective_date = doc.get(
                    "effective_date",
                    "Not found")
                expiry_date = doc.get(
                    "expiry_date",
                    "Not found")
                print(
                    f"  [Parser Agent] "
                    f"Parsed {len(chunks)} chunks "
                    f"· type: {contract_type}"
                )
            else:
                # Knowledge-only mode
                chunks = [{
                    "text": query,
                    "clause_type": "general",
                    "position": 0,
                    "char_count": len(query)
                }]
                contract_type = "unknown"
                jurisdiction = "US — General"
                effective_date = "Not found"
                expiry_date = "Not found"
                print(
                    f"  [Parser Agent] "
                    f"No file — knowledge-only mode"
                )

            latency_ms = int((time.time() - start) * 1000)

            self.status = "done"

            result = {
                "agent_id": self.agent_id,
                "trace_id": self.trace_id,
                "status": "done",
                "contract_type": contract_type,
                "jurisdiction": jurisdiction,
                "effective_date": effective_date,
                "expiry_date": expiry_date,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "file_path": file_path or "none",
                "error": None,
                "latency_ms": latency_ms
            }

            self._log(result)
            return result

        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            self.status = "failed"

            # Circuit breaker —
            # return empty chunks not a crash
            result = {
                "agent_id": self.agent_id,
                "trace_id": self.trace_id,
                "status": "failed",
                "contract_type": "unknown",
                "jurisdiction": "US — General",
                "effective_date": "Not found",
                "expiry_date": "Not found",
                "chunks": [],
                "chunk_count": 0,
                "file_path": file_path or "none",
                "error": str(e),
                "latency_ms": latency_ms
            }

            print(
                f"  [Parser Agent] "
                f"FAILED: {e} — "
                f"continuing without document"
            )

            self._log(result)
            return result

    def _log(self, result: dict) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(
            self.log_dir,
            f"parser_{self.trace_id}.json")
        # Remove chunks from log to keep it small
        log_data = {k: v for k, v in result.items() if k != "chunks"}
        log_data["chunk_count"] = result.get("chunk_count", 0)
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
