import os
import re
import json
import time
import hashlib
import argparse
import numpy as np
from datetime import datetime

from agents.orchestrator import Orchestrator
from legal_intelligence_agent.text_signals import any_signal
from legal_intelligence_agent.confidence import (
    drop_counts,
    partition_by_threshold,
    passes_threshold,
)


class MockLLMClient:
    def __init__(self):
        self.dim = 384
        self.stopwords = {
            "the", "and", "or", "a", "an", "to", "of", "in", "for", "with", "on", "by",
            "shall", "will", "may", "must", "is", "are", "be", "as", "it", "this", "that",
            "between", "within", "without", "any", "all", "such", "its", "their", "party", "parties",
        }

    def _hash_token_to_index(self, token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim

    def embed(self, text: str) -> np.ndarray:
        tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        tokens = [t for t in tokens if len(t) >= 3 and t not in self.stopwords]
        vec = np.zeros((self.dim,), dtype="float32")
        for tok in tokens:
            vec[self._hash_token_to_index(tok)] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec if norm == 0.0 else (vec / norm).astype("float32")

    def classify(self, query: str) -> dict:
        q = query.strip().lower()

        # Only flag as vague if truly vague
        # 1-2 words with no legal context
        words = q.split()
        truly_vague = (
            len(words) <= 2 and
            not any_signal(q, [
                "contract", "agreement", "clause",
                "terminate", "payment", "liable",
                "confidential", "ip", "license",
                "employ", "breach", "notice",
                "obligation", "own", "resign",
                "sign", "legal", "law", "term",
            ])
        )

        if truly_vague:
            return {
                "clarification_needed": True,
                "suggested_topics": [
                    "Termination and notice periods",
                    "Payment terms and late fees",
                    "Confidentiality obligations",
                ],
                "intent": "",
                "confidence": 0.5,
                "contract_type": "unknown",
                "entities": {
                    "time_references": [],
                    "monetary_amounts": [],
                    "party_references": [],
                },
                "raw_query": query,
            }

        # Intent detection
        if any_signal(q, [
            "terminat", "resign", "notice",
            "end", "cancel", "exit",
        ]):
            intent = "termination"
        elif any_signal(q, [
            "breach", "violat", "break",
            "fail", "default",
        ]):
            intent = "breach"
        elif any_signal(q, [
            "payment", "pay", "fee", "invoice",
            "cost", "price", "royalt",
        ]):
            intent = "payment"
        elif any_signal(q, [
            "data", "privacy", "personal",
            "gdpr", "ccpa", "breach",
        ]):
            intent = "data"
        elif any_signal(q, [
            "ip", "intellectual", "property",
            "own", "copyright", "patent",
            "trademark", "license", "licens",
        ]):
            intent = "IP"
        elif any_signal(q, [
            "employ", "employee", "employer",
            "hire", "fire", "salary", "wage",
            "non-compete", "non-solicit",
        ]):
            intent = "employment"
        elif any_signal(q, [
            "confidential", "secret", "disclose",
            "nda", "non-disclosure",
        ]):
            intent = "confidentiality"
        else:
            intent = "liability"

        return {
            "clarification_needed": False,
            "intent": intent,
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

    def analyze(self, chunks: list, query: str, intent: str) -> list:
        out = []
        for chunk in chunks:
            text = str(chunk.get("text", ""))
            chunk_text = text

            termination_signals = [
                "without cause",
                "severance",
                "term end date",
                "termination benefit",
                "accrued benefits",
                "lump sum",
                "fired",
                "dismiss",
                "layoff",
                "notice of non-renewal",
                "good reason",
            ]

            severance_signals = [
                "severance payment",
                "severance pay",
                "continuation pay",
                "termination benefit",
                "eip annual award",
                "third special award",
                "release of claims",
                "pro rata basis",
            ]

            text_lower = chunk_text.lower()

            clause_type = str(chunk.get("clause_type", "general"))
            if any(s in text_lower
                   for s in severance_signals):
                clause_type = "severance"
            elif any(s in text_lower
                     for s in termination_signals):
                clause_type = "termination"

            ct = clause_type.lower()
            if ct in {"termination", "breach"}:
                risk = "CRITICAL"
            elif ct in {"liability", "payment"}:
                risk = "HIGH"
            elif ct in {"access_suspension"}:
                risk = "HIGH"
            elif ct in {"service_level"}:
                risk = "MEDIUM"
            elif ct in {"confidentiality", "ip"}:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            if clause_type in {
                    "termination", "severance"}:
                risk = "HIGH"
            triggered = intent.lower() in text.lower() or intent.lower() in ct
            # Vary confidence based on how
            # strongly this clause matches
            # the intent / query, so the UI
            # doesn't show identical scores
            # for every clause.
            query_lower = query.lower()
            query_hits = sum(
                1 for w in query_lower.split()
                if len(w) > 3 and w in text_lower)
            intent_in_type = (
                intent.lower() in ct
                if intent else False)
            if intent_in_type and query_hits >= 2:
                conf = 0.94
            elif intent_in_type or query_hits >= 3:
                conf = 0.88
            elif triggered:
                conf = 0.82
            elif query_hits >= 1:
                conf = 0.75
            else:
                conf = 0.68
            out.append(
                {
                    "text": text,
                    "clause_type": clause_type,
                    "position": chunk.get("position", 0),
                    "risk_level": risk,
                    "legal_meaning": "This clause governs " + clause_type,
                    "triggered": bool(triggered),
                    "source_position": chunk.get("position", 0),
                    "confidence": conf,
                }
            )
        return out

    def translate(self, analyzed):
        jargon = {
            "hereinafter": "from now on",
            "pursuant": "according to",
            "notwithstanding": "despite",
            "whereas": "given that",
        }
        if isinstance(analyzed, list):
            out = []
            for clause in analyzed:
                text = str(clause.get("text", ""))
                for old, new in jargon.items():
                    text = re.sub(rf"\b{old}\b", new, text, flags=re.IGNORECASE)
                c = dict(clause)
                c["plain_english"] = "In plain English: " + text
                out.append(c)
            return out
        text = str(analyzed.get("text", ""))
        for old, new in jargon.items():
            text = re.sub(rf"\b{old}\b", new, text, flags=re.IGNORECASE)
        return "In plain English: " + text

    def gap_check(self, analyzed: list, contract_type: str) -> list:
        ct = (contract_type or "").lower()
        if ct == "nda":
            missing = ["governing_law", "return_of_materials"]
        elif ct in {"vendor", "supplier"}:
            missing = ["data_protection", "dispute_resolution"]
        elif ct == "employment":
            missing = ["non_solicitation", "ip_assignment"]
        elif ct == "saas":
            missing = ["sla", "data_processing"]
        else:
            missing = ["dispute_resolution"]
        return [
            {"clause_type": x, "severity": "HIGH", "description": "No " + x + " clause found."}
            for x in missing
        ]

    def validate(self, analyzed: list, translated: list, gaps: list) -> dict:
        translated_by_pos = {int(c.get("source_position", c.get("position", -1))): c for c in translated}
        kept, dropped_missing, dropped_low = partition_by_threshold(analyzed)
        final = []
        for clause in kept:
            pos = int(clause.get("source_position", clause.get("position", -1)))
            merged = dict(clause)
            merged.update(translated_by_pos.get(pos, {}))
            final.append(merged)
        critical_count = sum(1 for c in final if c.get("risk_level") == "CRITICAL")
        high_count = sum(1 for c in final if c.get("risk_level") == "HIGH")
        return {
            "analyzed_clauses": final,
            "missing_clauses": gaps,
            "summary": f"Found {len(final)} clauses. {critical_count} are CRITICAL.",
            "total_clauses_found": len(final),
            "total_missing": len(gaps),
            "critical_count": critical_count,
            "high_count": high_count,
            **drop_counts(dropped_missing, dropped_low),
        }


class ClaudeLLMClient:
    def __init__(self, api_key: str = None):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-4-5"
        self.prompt_version = "v1.0.0"
        self.total_tokens = 0
        self.total_cost = 0.0
        self._system_prompt = self._load_system()

    def _load_system(self) -> str:
        try:
            with open("prompts/v1.0.0_system.txt", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return (
                "You are a Legal Intelligence Agent. "
                "Analyze contracts and return "
                "structured JSON only. "
                "No markdown. No explanation."
            )

    def _call(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self._system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = input_tokens / 1_000_000 * 3.00 + output_tokens / 1_000_000 * 15.00
        self.total_tokens += input_tokens + output_tokens
        self.total_cost += cost
        return response.content[0].text

    def _parse_json(self, raw: str):
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"```(?:json)?\n?", "", raw).strip("`").strip()
        return json.loads(raw)

    def embed(self, text: str) -> np.ndarray:
        import hashlib as _hashlib

        dim = 384
        stopwords = {
            "the", "and", "or", "a", "an", "to", "of", "in", "for", "with", "on", "by",
            "shall", "will", "may", "must", "is", "are", "be", "as", "it", "this", "that",
            "party", "parties", "between", "within",
        }
        tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        tokens = [t for t in tokens if len(t) >= 3 and t not in stopwords]
        vec = np.zeros((dim,), dtype="float32")
        for tok in tokens:
            h = int(_hashlib.md5(tok.encode()).hexdigest(), 16) % dim
            vec[h] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec
        return (vec / norm).astype("float32")

    def classify(self, query: str) -> dict:
        prompt = f"""
Classify this legal query. Return JSON only.
No explanation. No markdown. Just JSON.

Query: "{query}"

Return this exact structure:
{{
  "clarification_needed": false,
  "intent": "termination or breach or payment or data or IP or employment or liability or confidentiality",
  "confidence": 0.85,
  "contract_type": "unknown",
  "entities": {{
    "time_references": [],
    "monetary_amounts": [],
    "party_references": []
  }},
  "suggested_topics": [],
  "raw_query": "{query}"
}}

Rules:
- If query is 1-2 words or vague set clarification_needed to true
- Pick the single most relevant intent
- confidence between 0.7 and 0.99
- Extract time references like "30 days"
- Extract money amounts like "$500"
- Extract party names like "vendor" "employee"
"""
        try:
            raw = self._call(prompt)
            result = self._parse_json(raw)

            # Validate intent is in allowed list
            allowed_intents = {
                "termination", "breach",
                "payment", "data", "IP",
                "employment", "liability",
                "confidentiality",
            }

            # Fix intent if invalid
            intent = result.get("intent", "")
            if intent not in allowed_intents:
                # Map common variations
                intent_map = {
                    "legal_advice": "liability",
                    "advice": "liability",
                    "general": "liability",
                    "unknown": "liability",
                    "contract": "liability",
                    "other": "liability",
                    "": "liability",
                }
                result["intent"] = intent_map.get(intent.lower(), "liability")

            # Validate confidence is float 0-1
            confidence = result.get("confidence", 0.85)
            try:
                confidence = float(confidence)
                if confidence < 0 or confidence > 1:
                    confidence = 0.85
            except Exception:
                confidence = 0.85
            result["confidence"] = confidence

            # Ensure all required keys exist
            result.setdefault("clarification_needed", False)
            result.setdefault("intent", "liability")
            result.setdefault("confidence", 0.85)
            result.setdefault("contract_type", "unknown")
            result.setdefault("entities", {
                "time_references": [],
                "monetary_amounts": [],
                "party_references": [],
            })
            result.setdefault("suggested_topics", [])
            result.setdefault("raw_query", "")

            return result

        except Exception:
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

    def analyze(
            self, chunks: list,
            query: str,
            intent: str,
            jurisdiction="US — General",
            effective_date="Not found",
            expiry_date="Not found") -> list:

        chunks_text = "\n\n".join([
            f"[Chunk {c['position']}]"
            f"({c['clause_type']}): "
            f"{c['text']}"
            for c in chunks
        ])
        import datetime as _dt
        _today = _dt.date.today().strftime(
            "%B %d, %Y")

        prompt = f"""
IMPORTANT RISK LEVEL RULES —
follow these exactly:
- Use CRITICAL when the clause
  directly answers the query AND
  involves serious consequences
  like: breach remedies, court
  injunctions, termination of
  rights, liability for damages,
  irreparable harm language.
- Use HIGH when the clause is
  directly on-topic for the query
  (e.g. query about confidentiality
  → confidentiality obligation
  clause = HIGH).
- Use MEDIUM when the clause is
  related but secondary.
- Use LOW when barely relevant.
- You MUST use at least one
  CRITICAL or HIGH in every
  response if any clause directly
  addresses the query topic.
- NEVER return MEDIUM for a clause
  that directly answers the query.

Analyze these contract clauses for the user's query.
Return a JSON array only. No explanation. No markdown.

Query: "{query}"
Intent: "{intent}"
Jurisdiction: "{jurisdiction}"
Contract effective date: "{effective_date}"
Contract expiry date: "{expiry_date}"
Today's date: "{_today}"

If the contract expiry date has
already passed before today, flag
any duration or termination clauses
as CRITICAL and note in plain_english
that this contract may no longer
be active.

If a specific obligation deadline
has already passed, note this in
the plain_english explanation.

Apply the law of this jurisdiction
when assessing risk levels. For
example, California has stronger
employee protections than Delaware.
A non-compete clause in California
is unenforceable — flag it CRITICAL.

Clauses:
{chunks_text}

For each clause return exactly:
{{
  "position": chunk_position_number_from_label,
  "plain_english": "one sentence plain explanation of what this clause does for the user",
  "risk_level": "CRITICAL if this clause directly answers the query AND has serious legal consequences (e.g. breach remedies, termination rights, liability caps). HIGH if this clause is directly relevant to the query topic. MEDIUM if this clause is related but not the main answer. LOW if this clause is only tangentially related.",
  "risk_score": a decimal from 0.50 to 0.99 representing how directly this clause answers the query. Use 0.90-0.99 for clauses that directly answer the query. Use 0.75-0.89 for clauses that are relevant but not the direct answer. Use 0.50-0.74 for clauses that are only tangentially related. Never give the same score to more than 3 clauses,
  "triggered": true if this clause directly answers the query, false otherwise
}}

Do NOT include clause_type in your response.
Do NOT default every score to the same value — vary scores based on relevance.
Lower the risk_level for clauses that mention the topic only in passing.

Remember: vary your risk_level
and risk_score values. If all
clauses get MEDIUM you have
failed this task.
"""
        try:
            raw = self._call(
                prompt, max_tokens=2048)
            parsed = self._parse_json(raw)
            if isinstance(parsed, list):
                fixed = []
                for item in parsed:
                    if isinstance(item, str):
                        fixed.append({
                            "text": item,
                            "clause_type": "general",
                            "position": len(fixed),
                            "risk_level": "MEDIUM",
                            "legal_meaning": item[:100],
                            "triggered": False,
                            "source_position": len(fixed),
                            "confidence": 0.65,
                        })
                    elif isinstance(item, dict):
                        pe = item.pop(
                            "plain_english", None)
                        if pe is not None:
                            item["legal_meaning"] = pe
                        rs = item.pop(
                            "risk_score", None)
                        if rs is not None:
                            try:
                                item["confidence"] = float(rs)
                            except (TypeError, ValueError):
                                item["confidence"] = 0.7
                        fixed.append(item)

                # Match Claude results back to
                # chunks by position — not index.
                # Build a lookup keyed by the
                # chunk's position so the output
                # is robust if Claude reorders or
                # drops items.
                chunk_by_pos = {
                    c.get("position", i): c
                    for i, c in enumerate(chunks)
                }
                # Deduplicate by position — but
                # reassign orphaned results to
                # unused chunk positions so we
                # don't lose clauses
                seen_positions = {}
                orphans = []
                for item in fixed:
                    pos = item.get("position")
                    if pos is None or pos not in chunk_by_pos:
                        orphans.append(item)
                        continue
                    if pos not in seen_positions:
                        seen_positions[pos] = item
                    else:
                        existing = seen_positions[pos]
                        existing_conf = existing.get(
                            "confidence",
                            existing.get(
                                "risk_score", 0))
                        new_conf = item.get(
                            "confidence",
                            item.get(
                                "risk_score", 0))
                        if new_conf > existing_conf:
                            orphans.append(existing)
                            seen_positions[pos] = item
                        else:
                            orphans.append(item)

                # Assign orphans to unused positions
                used_positions = set(
                    seen_positions.keys())
                unused_positions = [
                    p for p in chunk_by_pos.keys()
                    if p not in used_positions]

                for orphan, pos in zip(
                        orphans, unused_positions):
                    orphan["position"] = pos
                    seen_positions[pos] = orphan

                fixed = list(seen_positions.values())
                for i, item in enumerate(fixed):
                    pos = item.get("position")
                    if pos is None and i < len(chunks):
                        pos = chunks[i].get(
                            "position", i)
                        item["position"] = pos
                    ch = chunk_by_pos.get(
                        pos,
                        chunks[i] if i < len(chunks)
                        else {})
                    # Always restore parser
                    # clause_type — never trust
                    # the LLM
                    item["clause_type"] = ch.get(
                        "clause_type",
                        item.get(
                            "clause_type",
                            "general"))
                    if not item.get("text"):
                        item["text"] = ch.get(
                            "text", "")
                    if item.get(
                            "source_position"
                    ) is None:
                        item["source_position"] = (
                            pos if pos is not None
                            else i)
                    item.setdefault(
                        "triggered", False)
                    # If Claude gave no
                    # confidence, derive a
                    # default from risk_level
                    if "confidence" not in item:
                        rl = item.get(
                            "risk_level", "MEDIUM")
                        item["confidence"] = {
                            "CRITICAL": 0.95,
                            "HIGH": 0.88,
                            "MEDIUM": 0.78,
                            "LOW": 0.65,
                        }.get(rl, 0.75)

                # Risk-level adjustments — only
                # bump up when Claude
                # underestimated, never force
                # everything to HIGH. Only the
                # clauses that are ALSO triggered
                # for this query get bumped.
                for item in fixed:
                    ct = item.get(
                        "clause_type", "")
                    rl = item.get(
                        "risk_level", "MEDIUM")
                    triggered = item.get(
                        "triggered", False)
                    if (triggered and
                            ct in {
                                "severance",
                                "termination",
                                "access_suspension"}
                            and rl == "LOW"):
                        item["risk_level"] = (
                            "MEDIUM")

                return fixed
            return parsed

        except Exception:
            return [{
                "text": c["text"],
                "clause_type": c[
                    "clause_type"],
                "position": c["position"],
                "risk_level": "MEDIUM",
                "legal_meaning": (
                    c["text"][:100]),
                "triggered": False,
                "source_position": c[
                    "position"],
                "confidence": 0.65,
            } for c in chunks]

    def translate(self, analyzed: list,
                  jurisdiction:
                  str = "US — General",
                  effective_date:
                  str = "Not found",
                  expiry_date:
                  str = "Not found"
                  ) -> list:
        if not analyzed:
            return analyzed

        result = list(analyzed)

        # Build one prompt with all clauses
        clauses_lines = []
        for i, clause in enumerate(result):
            clause_text = str(
                clause.get("text", "")
            )[:800]
            clauses_lines.append(
                f"CLAUSE_{i}: {clause_text}")
        clauses_text = "\n".join(clauses_lines)
        import datetime as _dt
        _today = _dt.date.today().strftime(
            "%B %d, %Y")

        prompt = (
            f"Contract jurisdiction: {jurisdiction}. "
            f"Contract effective date: {effective_date}. "
            f"Contract expiry date: {expiry_date}. "
            f"Today's date: {_today}. "
            f"IMPORTANT: If the contract expiry "
            f"date has already passed before "
            f"today, add this note ONLY to "
            f"clauses that explicitly mention "
            f"the contract end date, term "
            f"duration, or expiry date: "
            f"Note: This contract expired on "
            f"[expiry date] — verify whether "
            f"it is still in effect. "
            f"Do NOT add this note to payment, "
            f"severance, death, disability, or "
            f"other clauses that do not directly "
            f"reference the contract term or "
            f"end date. "
            f"Rewrite each clause in plain English. "
            f"Do NOT include the original legal text. "
            f"Do NOT include section numbers. "
            f"Do NOT copy any text from the original. "
            f"Write only a plain English explanation a non-lawyer would understand. "
            f"No legal jargon. "
            f"You MUST provide a plain_english translation for EVERY clause in the input array — no exceptions. "
            f"Never return the original legal text as the translation. "
            f"Never skip a clause. "
            f"Never return an empty string. "
            f"If a clause is a formal amendment like \"10. Amendment to Section 8.E...\" — translate what that amendment actually says in plain English, not the amendment header. "
            f"For very long clauses that are formal amendments (starting with a number like \"10. Amendment\"), summarize the SUBSTANCE of what the amendment changes — not the amendment header. Focus on what it means for the employee in plain English. Write 2-3 sentences maximum per clause. "
            f"Always write in second person (you/your). Never use he/she/his/her or the person's name. "
            f"Be thorough — each translation should be 2-4 sentences covering all the key points of the clause.\n\n"
            f"For each clause return one line:\n"
            f"CLAUSE_0: plain English here\n"
            f"CLAUSE_1: plain English here\n\n"
            f"Clauses:\n{clauses_text}\n\n"
            f"Return ONLY the CLAUSE_N: lines. "
            f"Nothing else."
        )

        try:
            raw = self._call(prompt, max_tokens=4096)
            lines = raw.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("CLAUSE_"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        try:
                            idx = int(parts[0].replace("CLAUSE_", ""))
                            if idx < len(result):
                                result[idx]["plain_english"] = parts[1].strip()
                        except ValueError:
                            pass
        except Exception:
            pass

        # Fill any missing translations
        for c in result:
            if "plain_english" not in c:
                c["plain_english"] = c.get("text", "")

        return result

    def gap_check(self, analyzed: list, contract_type: str) -> list:
        found_types = list(set([c["clause_type"] for c in analyzed]))
        prompt = f"""
Check this contract for missing required clauses.
Return a JSON array only.
No explanation. No markdown.

Contract type: "{contract_type}"
Clause types found: {json.dumps(found_types)}

Do NOT flag a clause as missing if a
similar clause type was already found.
For example if confidentiality_use_restriction
was found do not flag confidentiality as missing.
Only flag clauses that are genuinely absent
from the contract entirely.
"""
        try:
            raw = self._call(prompt)
            gaps = self._parse_json(raw)
            validated_gaps = []
            valid_severities = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
            for gap in gaps:
                if not isinstance(gap, dict):
                    continue
                severity = str(gap.get("severity", "MEDIUM")).upper()
                if severity not in valid_severities:
                    severity = "MEDIUM"
                gap["severity"] = severity
                if gap.get("clause_type"):
                    validated_gaps.append(gap)
            return validated_gaps
        except Exception:
            return []

    def validate(self, analyzed: list, translated: list, gaps: list) -> dict:
        # Was c.get("confidence", 1.0) >= 0.6, so a clause the model never
        # scored defaulted to full confidence and passed the gate.
        filtered, dropped_missing, dropped_low = partition_by_threshold(translated)
        critical = sum(1 for c in filtered if c.get("risk_level") == "CRITICAL")
        high = sum(1 for c in filtered if c.get("risk_level") == "HIGH")
        summary = f"Found {len(filtered)} clauses. {critical} are CRITICAL."
        if critical == 0 and high == 0:
            summary += " No high-risk clauses detected."
        return {
            "analyzed_clauses": filtered,
            "missing_clauses": gaps,
            "summary": summary,
            "total_clauses_found": len(filtered),
            "total_missing": len(gaps),
            "critical_count": critical,
            "high_count": high,
            **drop_counts(dropped_missing, dropped_low),
        }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Legal Intelligence Agent v1.0.0"
            " — Multi-Agent Mode"))
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Legal question to ask")
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to contract file")
    parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "markdown", "dict"],
        help="Output format")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM — no API key needed")
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Enable human approval gate")
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version")
    args = parser.parse_args()

    if args.version:
        print(
            "Legal Intelligence Agent "
            "v1.0.0 — Multi-Agent Mode")
        return

    # Header
    print()
    print("╔══════════════════════════════════════╗")
    print("║  Legal Intelligence Agent  v1.0.0   ║")
    print("║  Multi-Agent Parallel Mode           ║")
    print("║  Type 'exit' to quit                 ║")
    print("╚══════════════════════════════════════╝")
    print()

    # Get query and file
    if args.query:
        query = args.query
        file_path = args.file
    else:
        file_path = input(
            "Contract file path "
            "(press Enter to skip): ").strip()
        if file_path.lower() == "exit":
            return
        file_path = (
            file_path if file_path else None)
        query = input(
            "Your legal question: ").strip()
        if query.lower() == "exit":
            return

    if not query:
        print("No question provided. Exiting.")
        return

    print()

    # Set up LLM client
    if args.mock:
        llm = MockLLMClient()
        print("Mode: MOCK (no API calls)")
    else:
        api_key = os.environ.get(
            "ANTHROPIC_API_KEY")
        if not api_key:
            print(
                "ERROR: ANTHROPIC_API_KEY "
                "not set.")
            print(
                "Run: "
                "$env:ANTHROPIC_API_KEY="
                "'your_key_here'")
            return
        llm = ClaudeLLMClient(
            api_key=api_key)
        print("Mode: CLAUDE (real LLM)")

    print()

    try:
        # Run orchestrator
        orchestrator = Orchestrator(
            llm_client=llm,
            log_dir="logs",
            output_format=args.format)

        result = orchestrator.run(
            query=query,
            file_path=file_path)

        if not result:
            return

        # Print output
        print()
        print("=" * 60)
        output = result.get("output", "")
        if args.format == "dict":
            print(json.dumps(
                output, indent=2,
                default=str))
        else:
            print(output)

        # Approval gate if requested
        if args.approve and result.get(
                "agent_result"):
            from approval_gate import (
                ApprovalGate)
            gate = ApprovalGate()
            approved = gate.review(
                result["agent_result"])
            print()
            print("=" * 60)
            print("POST-APPROVAL REPORT")
            print("=" * 60)
            from legal_intelligence_agent\
                .output_report import (
                    OutputReport)
            reporter = OutputReport()
            final_output = reporter.generate(
                approved,
                format=args.format)
            print(final_output)

        # Print cost summary
        print()
        if not args.mock:
            print(
                f"Tokens:  "
                f"{result.get('total_tokens', 0):,}")
            print(
                f"Cost:    "
                f"${result.get('total_cost_usd', 0):.6f}"
                f" USD")
        print(
            f"Latency: "
            f"{result.get('total_latency_ms', 0)}ms")
        print(
            f"Mode:    "
            f"multi-agent parallel")
        print()

    except Exception as e:
        print(f"\nError: {e}")
        print(
            "Check your file path "
            "and try again.")


if __name__ == "__main__":
    main()
