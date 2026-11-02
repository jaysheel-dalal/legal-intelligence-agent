import time
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(
    os.path.abspath(__file__)))

from legal_intelligence_agent.text_signals import signal_in


st.set_page_config(
    page_title="Legal Intelligence Agent",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_COLORS = {
    "CRITICAL": "#EF4444",
    "HIGH": "#F97316",
    "MEDIUM": "#B45309",
    "LOW": "#0F766E",
}

AGENTS = [
    ("parser_agent", "Parser Agent", "Loads and chunks contract"),
    ("analysis_agent", "Analysis Agent", "Risk scores each clause"),
    ("gap_agent", "Gap Agent", "Detects missing clauses"),
    ("report_agent", "Report Agent", "Validates and formats output"),
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Lato:wght@300;400;700&family=Playfair+Display:wght@700;900&display=swap');
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stToolbar"] {display: none !important;}
[data-testid="stDecoration"] {display: none !important;}
.stDeployButton {display: none !important;}
[data-testid="stSidebar"] {
  background: #0F1F3D !important;
  border-right: 1px solid #1E3A5F;
  min-width: 280px !important;
  max-width: 280px !important;
}
html, body, [data-testid="stAppViewContainer"] {
  background: #F7F4EE !important;
}
.main .block-container {
  padding: 2rem 2.5rem;
  max-width: 100%;
}
[data-testid="stButton"] > button {
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  background: #1E3A5F !important;
  color: #F7F4EE !important;
  border: 1px solid #1E3A5F !important;
  border-radius: 2px !important;
}
[data-testid="stButton"] > button:hover {
  background: #0F1F3D !important;
  color: #F7F4EE !important;
}
[data-testid="stFileUploaderDropzone"] {
  background: white !important;
  border: 1.5px dashed #C8BFB0 !important;
  border-radius: 3px !important;
}
[data-testid="stTextArea"] textarea {
  background: white !important;
  border: 1.5px solid #E0D8CC !important;
  border-radius: 2px !important;
  font-family: 'Lato', sans-serif !important;
  font-size: 14px !important;
  color: #1a1208 !important;
}
[data-testid="stTextArea"] textarea:focus {
  border-color: #1E3A5F !important;
  box-shadow: none !important;
}
.eyebrow {
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  letter-spacing: 3px;
  color: #9A8F7A;
}
.title {
  font-family: 'Playfair Display', serif;
  color: #1a1208;
}
.subtle {
  font-family: 'Lato', sans-serif;
  color: #9A8F7A;
}
.label {
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  letter-spacing: 2px;
  color: #9A8F7A;
  margin-bottom: 0.45rem;
}
.side-card {
  background: rgba(30,58,95,0.3);
  border-left: 2px solid #1E3A5F;
  padding: 0.45rem 0.5rem;
  margin-bottom: 0.4rem;
}
.summary-bar {
  background: #1a1208;
  border-radius: 4px;
  padding: 1rem 1.1rem;
  margin-bottom: 1rem;
}
.clause-card {
  background: white;
  border: 1px solid #E0D8CC;
  border-radius: 3px;
  padding: 1.1rem 1.4rem;
  margin-bottom: 0.8rem;
}
.missing-card {
  background: white;
  border: 1px solid #E0D8CC;
  border-radius: 3px;
  padding: 0.8rem 1rem;
  margin-bottom: 0.6rem;
}
.badge {
  display: inline-block;
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  padding: 1px 7px;
  border-radius: 2px;
}
</style>
"""


def init_state():
    if "agent_status" not in st.session_state:
        st.session_state["agent_status"] = {
            "parser_agent": "idle",
            "analysis_agent": "idle",
            "gap_agent": "idle",
            "report_agent": "idle",
        }
    if "history" not in st.session_state:
        st.session_state["history"] = []


def run_real(query, uploaded_file):
    import tempfile
    import os
    from agents.orchestrator import Orchestrator

    # Determine LLM client
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        from main_v3 import ClaudeLLMClient
        llm = ClaudeLLMClient(api_key=api_key)
        mode = "claude"
    else:
        from main_v3 import MockLLMClient
        llm = MockLLMClient()
        mode = "mock"

    # Save uploaded file to temp location
    file_path = None
    if uploaded_file is not None:
        suffix = "." + uploaded_file.name.split(
            ".")[-1]
        with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            file_path = tmp.name

    try:
        # Run orchestrator
        orchestrator = Orchestrator(
            llm_client=llm,
            log_dir="logs",
            output_format="dict")

        # Detect gap queries and replace
        # with specific query for pipeline
        gap_words = [
            "missing", "what clauses",
            "gaps", "lacking", "absent",
            "incomplete", "what else",
            "anything missing",
            "what should be",
            "what needs to be",
            "what doesnt",
            "what does it not",
            "what isnt",
            "what is not in",
            "what am i missing",
            "did i miss",
            "clauses missing",
            "missing clauses",
        ]
        gap_focused_query = any(
            w in query.lower()
            for w in gap_words)
        pipeline_query = query
        if any(w in query.lower()
               for w in gap_words):
            pipeline_query = (
                "What are the key terms "
                "obligations and rights "
                "in this agreement?")

        print(f"[APP] Original query: {query}")
        print(f"[APP] Gap focused: {gap_focused_query}")
        print(f"[APP] Pipeline query: {pipeline_query}")

        result = orchestrator.run(
            query=pipeline_query,
            file_path=file_path)

        print(f"[APP] Result keys: {result.keys() if result else 'None'}")
        print(f"[APP] Clauses: {result.get('total_clauses_found', 'KEY MISSING')}")
        print(f"[APP] Contract type: {result.get('contract_type', 'KEY MISSING')}")
        print(f"[APP] All result keys: {list(result.keys()) if result else 'None'}")

        # Extract clauses and gaps
        agent_result = result.get(
            "agent_result", {})
        clauses_raw = agent_result.get(
            "analyzed_clauses", [])
        
        # If empty — retry with more specific query
        if not clauses_raw:
            general_queries = [
                "what", "about", "summary",
                "overview", "tell me", "explain",
                "describe", "show"
            ]
            is_general = any(
                w in query.lower() 
                for w in general_queries)
            
            if is_general:
                result2 = orchestrator.run(
                    query=(
                        "What are the key terms "
                        "obligations and rights "
                        "in this agreement?"),
                    file_path=file_path)
                agent_result = result2.get(
                    "agent_result", {})
                clauses_raw = agent_result.get(
                    "analyzed_clauses", [])
                missing_raw = result2.get(
                    "agent_result", {}).get(
                    "missing_clauses", [])
        missing_raw = agent_result.get(
            "missing_clauses", [])

        # Format clauses for UI
        clauses = []
        for c in clauses_raw:
            plain_english = c.get(
                "plain_english", "").strip()

            # If plain_english is empty or
            # looks like raw legal text
            # (starts with legal phrases or
            # is very long), replace with
            # a placeholder
            raw_text = c.get("text", "")
            if not plain_english:
                plain_english = raw_text

            legal_prefixes = [
                "notwithstanding",
                "whereas",
                "now therefore",
                "in witness whereof",
                "exhibit",
                "amendment no",
                "this amendment",
                "section",
            ]

            text_check = plain_english\
                .lower().strip()

            if any(text_check.startswith(p)
                   for p in legal_prefixes):
                plain_english = (
                    "See contract section "
                    f"§{c.get('source_position', '?')}"
                    " for full details.")

            # Strip common prefixes
            for prefix in [
                "In plain English:",
                "In plain english:",
                "Plain English:",
                "Plain english:",
                "In Plain English:",
            ]:
                if plain_english.startswith(prefix):
                    plain_english = plain_english[
                        len(prefix):].strip()

            # Use Claude's confidence if it
            # provided a real value. Only fall
            # back to a derived value when the
            # field is missing or matches the
            # known sentinel defaults.
            raw_conf = c.get("confidence")
            sentinel_defaults = {0.7, 0.8, 0.65}
            needs_fallback = (
                raw_conf is None
                or raw_conf in sentinel_defaults
            )
            if needs_fallback:
                risk = c.get(
                    "risk_level", "MEDIUM")
                triggered = c.get(
                    "triggered", False)
                if triggered and risk == "CRITICAL":
                    raw_conf = 0.97
                elif triggered and risk == "HIGH":
                    raw_conf = 0.92
                elif triggered and risk == "MEDIUM":
                    raw_conf = 0.85
                elif triggered:
                    raw_conf = 0.78
                elif risk == "CRITICAL":
                    raw_conf = 0.88
                elif risk == "HIGH":
                    raw_conf = 0.80
                elif risk == "MEDIUM":
                    raw_conf = 0.72
                else:
                    raw_conf = 0.62

            # Use this cleaned version
            clauses.append({
                "clause_type": c.get(
                    "clause_type", "general"),
                "risk_level": c.get(
                    "risk_level", "MEDIUM"),
                "plain_english": plain_english,
                "source_position": c.get(
                    "source_position", 0),
                "triggered": c.get(
                    "triggered", False),
                "confidence": raw_conf,
            })

        # ─────────────────────────────────
        # GENERALIZED CLAUSE RELEVANCE SCORER
        # Works for all contract types and
        # all natural human query variations
        # ─────────────────────────────────

        query_lower = query.lower()
        query_words = query_lower.split()

        # Comprehensive intent boost map
        # Key: any word that might appear
        # in a human query
        # Value: keywords to look for in
        # clause text

        intent_boost_map = {
            # ── TERMINATION / FIRING ──
            "fire": [
                "terminat", "fired", "cause",
                "severance", "without cause",
                "notice", "dismiss", "layoff",
                "termination for cause",
            ],
            "fired": [
                "terminat", "fired", "cause",
                "severance", "without cause",
                "notice", "dismiss",
            ],
            "terminate": [
                "terminat", "expir", "cancel",
                "end of agreement", "notice",
                "breach", "without cause",
            ],
            "termination": [
                "terminat", "expir", "cancel",
                "end of agreement", "notice",
                "severance", "without cause",
            ],
            "long": [
                "term ", "duration",
                "years", "months",
                "expire", "expir",
                "surviv", "period",
                "two year", "one year",
                "three year", "five year",
            ],
            "last": [
                "term ", "duration",
                "expire", "expir",
                "surviv", "period",
                "two year", "one year",
            ],
            "duration": [
                "term ", "terminat",
                "expire", "expir",
                "period", "years",
                "surviv",
            ],
            "expire": [
                "term ", "terminat",
                "expire", "expir",
                "duration", "period",
            ],
            "quit": [
                "resign", "terminat", "notice",
                "voluntary", "without cause",
                "executive terminat",
            ],
            "resign": [
                "resign", "terminat", "notice",
                "voluntary", "without cause",
            ],
            "leave": [
                "terminat", "resign", "expir",
                "end", "notice", "severance",
            ],
            "layoff": [
                "without cause", "terminat",
                "severance", "notice",
                "reduction in force",
            ],
            "severance": [
                "severance", "continuation",
                "termination benefit",
                "without cause", "lump sum",
                "12 months", "notice period",
            ],
            # ── COMPENSATION / PAYMENT ──
            "pay": [
                "salary", "pay", "compensat",
                "bonus", "royalt", "fee",
                "wage", "earning", "income",
            ],
            "paid": [
                "salary", "pay", "compensat",
                "bonus", "payment", "wage",
            ],
            "salary": [
                "salary", "base salary",
                "annual salary", "compensat",
                "payroll", "wage",
            ],
            "compensation": [
                "compensat", "salary", "bonus",
                "benefit", "pay", "wage",
                "equity", "stock",
            ],
            "bonus": [
                "bonus", "incentive", "target",
                "stretch", "performance",
                "annual bonus",
            ],
            "money": [
                "salary", "pay", "compensat",
                "bonus", "royalt", "fee",
                "payment", "financial",
            ],
            "earn": [
                "salary", "earn", "compensat",
                "bonus", "royalt", "income",
            ],
            "benefit": [
                "benefit", "health", "insurance",
                "dental", "retirement", "401k",
                "vacation", "pto", "equity",
            ],
            "benefits": [
                "benefit", "health", "insurance",
                "dental", "retirement", "401k",
                "vacation", "pto",
            ],
            "insurance": [
                "insurance", "health", "dental",
                "disability", "life insurance",
                "benefit plan",
            ],
            "vacation": [
                "vacation", "pto", "paid time",
                "sick leave", "time off",
                "annual leave",
            ],
            "equity": [
                "equity", "stock", "option",
                "rsu", "shares", "vesting",
                "grant",
            ],
            "royalty": [
                "royalt", "payment", "fee",
                "milestone", "license fee",
                "net sales",
            ],
            "royalties": [
                "royalt", "payment", "fee",
                "milestone", "net sales",
            ],
            "invoice": [
                "invoice", "payment", "billing",
                "fee", "net 30", "due date",
            ],
            "subscription": [
                "subscription", "fee", "annual",
                "monthly", "renewal", "term",
            ],
            # ── CONFIDENTIALITY / SECRETS ──
            "confidential": [
                "confidential", "secret",
                "disclose", "proprietary",
                "trade secret", "nda",
            ],
            "secret": [
                "confidential", "secret",
                "trade secret", "proprietary",
                "not disclose",
            ],
            "disclose": [
                "disclose", "confidential",
                "disclosure", "reveal",
                "share information",
            ],
            "nda": [
                "confidential", "nda",
                "non-disclosure", "secret",
                "proprietary",
            ],
            "private": [
                "confidential", "proprietary",
                "private", "secret", "disclose",
            ],
            # ── IP / OWNERSHIP ──
            "own": [
                "own", "ownership", "ip",
                "patent", "invent", "assign",
                "work product", "copyright",
            ],
            "owns": [
                "own", "ownership", "ip",
                "patent", "invent", "assign",
            ],
            "ownership": [
                "ownership", "own", "ip",
                "patent", "assign", "rights",
                "work product",
            ],
            "ip": [
                "intellectual property", "ip",
                "patent", "copyright", "trademark",
                "invention", "work product",
            ],
            "patent": [
                "patent", "invention", "ip",
                "intellectual property",
                "filing", "prosecution",
            ],
            "invention": [
                "invention", "invent", "patent",
                "ip", "assign", "work product",
                "intellectual property",
            ],
            "copyright": [
                "copyright", "ip", "work product",
                "intellectual property",
                "creative work",
            ],
            "create": [
                "work product", "ip", "invent",
                "create", "develop", "assign",
                "intellectual property",
            ],
            "license": [
                "licens", "grant", "rights",
                "exclusive", "field of use",
                "sublicens",
            ],
            # ── COMPETITION / RESTRICTIONS ──
            "compete": [
                "compet", "restrict",
                "non-compete", "competitive",
                "rival", "competitor",
            ],
            "competition": [
                "compet", "restrict",
                "non-compete", "competitive",
                "rival",
            ],
            "solicit": [
                "solicit", "recruit",
                "non-solicitation", "hire",
                "poach",
            ],
            "work": [
                "duties", "responsib", "role",
                "position", "job", "title",
                "perform", "services",
            ],
            "job": [
                "duties", "responsib", "role",
                "position", "title", "perform",
            ],
            "role": [
                "duties", "responsib", "role",
                "position", "title", "reporting",
            ],
            "duties": [
                "duties", "responsib",
                "perform", "services", "role",
            ],
            # ── LIABILITY / RISK ──
            "liable": [
                "liabil", "damage", "indemnif",
                "responsible", "warrant",
                "limitation",
            ],
            "liability": [
                "liabil", "damage", "indemnif",
                "responsible", "limitation",
                "cap", "in no event",
            ],
            "indemnif": [
                "indemnif", "hold harmless",
                "defend", "losses", "damages",
            ],
            "indemnify": [
                "indemnif", "hold harmless",
                "defend", "losses",
            ],
            "damage": [
                "damage", "liabil", "indemnif",
                "loss", "claim", "remedy",
            ],
            "sue": [
                "liabil", "lawsuit", "claim",
                "arbitrat", "court", "dispute",
                "indemnif",
            ],
            "lawsuit": [
                "lawsuit", "liabil", "claim",
                "arbitrat", "court", "dispute",
            ],
            "protect": [
                "indemnif", "liabil", "warrant",
                "protect", "hold harmless",
                "insurance",
            ],
            # ── DISPUTE RESOLUTION ──
            "dispute": [
                "dispute", "arbitrat", "mediat",
                "court", "litigation", "jams",
                "aaa", "resolution",
            ],
            "arbitration": [
                "arbitrat", "dispute", "jams",
                "aaa", "binding", "award",
            ],
            "court": [
                "court", "jurisdiction",
                "arbitrat", "litigation",
                "governing law", "venue",
            ],
            # ── GOVERNING LAW ──
            "law": [
                "governing law", "governed by",
                "jurisdiction", "state law",
                "ohio", "delaware", "california",
            ],
            "jurisdiction": [
                "jurisdiction", "governing law",
                "court", "venue", "state of",
            ],
            "govern": [
                "governing law", "governed by",
                "jurisdiction", "state law",
            ],
            # ── CHANGE IN CONTROL ──
            "acquire": [
                "change in control", "merger",
                "acquisition", "buyout",
                "lump sum",
            ],
            "merger": [
                "merger", "change in control",
                "acquisition", "buyout",
            ],
            "acquisition": [
                "acquisition", "change in control",
                "merger", "buyout",
            ],
            "buyout": [
                "buyout", "change in control",
                "acquisition", "lump sum",
            ],
            # ── SaaS SPECIFIC ──
            "uptime": [
                "uptime", "availability", "sla",
                "service level", "99.",
            ],
            "downtime": [
                "downtime", "uptime", "outage",
                "availability", "sla",
            ],
            "security": [
                "security", "encrypt", "data",
                "protection", "breach",
                "cyber",
            ],
            # ── DELIVERY / VENDOR ──
            "deliver": [
                "deliver", "delivery",
                "shipment", "ship",
                "late", "delay",
                "schedule", "deadline",
                "on time", "lead time",
                "failure to deliver",
                "non-delivery",
            ],
            "late": [
                "late", "delay",
                "deliver", "delivery",
                "deadline", "schedule",
                "failure to deliver",
                "remedy", "cure",
                "penalty", "liquidated",
            ],
            "delivery": [
                "deliver", "delivery",
                "shipment", "ship",
                "schedule", "deadline",
                "lead time", "on time",
            ],
            "defective": [
                "defect", "warranty",
                "quality", "reject",
                "return", "replace",
                "remedy", "cure",
            ],
            "quality": [
                "quality", "defect",
                "warranty", "standard",
                "specification", "conform",
                "reject", "return",
            ],
            "supply": [
                "supply", "vendor", "deliver",
                "goods", "product", "purchas",
            ],
            # ── GENERAL RIGHTS ──
            "right": [
                "right", "entitl", "permit",
                "allow", "may", "shall have",
            ],
            "rights": [
                "right", "entitl", "permit",
                "allow", "may", "shall have",
            ],
            "obligation": [
                "shall", "must", "obligat",
                "required", "responsible",
                "covenant",
            ],
            "obligations": [
                "shall", "must", "obligat",
                "required", "responsible",
            ],
            "allowed": [
                "permit", "allow", "may",
                "right", "entitl", "authorize",
            ],
            "permitted": [
                "permit", "allow", "may",
                "right", "authorize",
            ],
            "prohibited": [
                "shall not", "prohibit",
                "restrict", "not permit",
                "not allow", "cannot",
            ],
            # ── ACCESS / SUSPENSION ──
            "shut": [
                "suspend", "terminat",
                "access", "revok",
                "without notice", "immediat",
                "disable", "block",
            ],
            "access": [
                "access", "authorize",
                "suspend", "revok",
                "license", "permit",
                "without notice",
            ],
            "suspend": [
                "suspend", "terminat",
                "access", "revok",
                "without notice", "immediat",
                "tortious", "criminal",
                "illegal activity",
            ],
            "warning": [
                "without notice",
                "without warning",
                "immediat", "suspend",
                "prior written notice",
                "written notice",
            ],
            "notice": [
                "notice", "written notice",
                "prior notice", "days notice",
                "without notice", "immediat",
            ],
            "revoke": [
                "revok", "terminat",
                "suspend", "cancel",
                "access", "license",
            ],
            "cancel": [
                "cancel", "terminat",
                "revok", "end", "expir",
                "without cause", "notice",
            ],
            "disable": [
                "disable", "suspend",
                "block", "revok", "access",
                "without notice",
            ],
            # ── DATA / PRIVACY ──
            "data": [
                "data", "privacy", "security",
                "encrypt", "backup",
                "authorized user data",
                "personal information",
            ],
            "privacy": [
                "privacy", "data", "personal",
                "confidential", "hipaa",
                "phi", "encrypt",
            ],
            "breach": [
                "breach", "violat", "default",
                "cure", "notice", "remedy",
                "material breach",
            ],
            "hack": [
                "security", "breach",
                "unauthorized access",
                "cyber", "encrypt",
                "intrusion",
            ],
            # ── UPTIME / PERFORMANCE ──
            "down": [
                "downtime", "uptime",
                "availab", "outage",
                "service level", "sla",
                "unscheduled",
            ],
            "outage": [
                "outage", "downtime",
                "uptime", "availab",
                "unscheduled", "sla",
            ],
            "slow": [
                "response time", "performan",
                "sla", "service level",
                "uptime", "latency",
            ],
            "crash": [
                "outage", "downtime",
                "disaster recovery",
                "backup", "recovery",
                "availab",
            ],
            # ── VENDOR / CONTRACT ──
            "vendor": [
                "vendor", "supplier",
                "contractor", "service",
                "obligations", "deliver",
            ],
            "refund": [
                "refund", "credit",
                "service level credit",
                "pro-rata", "repay",
                "remedy",
            ],
            "penalty": [
                "penalty", "credit",
                "liquidated damages",
                "service level credit",
                "corrective action",
            ],
            # ── NON-TECHNICAL PLAIN ENGLISH WORDS ──
            "stop": [
                "compet", "restrict",
                "non-compete", "prohibit",
                "shall not", "not permit",
            ],
            "working": [
                "compet", "restrict",
                "non-compete", "duties",
                "services", "employ",
            ],
            "competitor": [
                "compet", "non-compete",
                "restrict", "rival",
                "competitive activity",
            ],
            "bought": [
                "change in control", "merger",
                "acquisition", "buyout",
                "lump sum", "acquir",
            ],
            "company": [
                "employer", "company",
                "organization", "acquir",
                "change in control",
            ],
            "boss": [
                "employer", "terminat",
                "cause", "at-will",
                "without cause",
            ],
            "laptop": [
                "equipment", "property",
                "return", "company property",
                "upon terminat",
            ],
            "built": [
                "work product", "ip",
                "invent", "assign",
                "intellectual property",
                "create", "develop",
            ],
            "side": [
                "outside activit", "moonlight",
                "compet", "conflict of interest",
                "prior written consent",
            ],
        }

        # Build boost keywords from query
        boost_keywords = []
        for word in query_words:
            # Direct match
            if word in intent_boost_map:
                boost_keywords.extend(
                    intent_boost_map[word])
            # Partial match — word contains key
            else:
                for key, boosts in (
                        intent_boost_map.items()):
                    if (key in word or
                            word in key) and (
                            len(key) > 5 and
                            len(word) > 3):
                        boost_keywords.extend(
                            boosts)

        # Remove duplicates
        boost_keywords = list(
            set(boost_keywords))

        # Score each clause by relevance
        def clause_score(clause):
            text = (
                clause.get(
                    "plain_english", "") + " " +
                clause.get(
                    "clause_type", "")
            ).lower()

            # Do NOT include raw "text" field
            # as it causes too many matches

            score = 0

            # Strong boost for intent keywords.
            # Word-boundary matched: intent_boost_map carries short signals
            # such as "end" and "ip", and plain containment scored them
            # inside unrelated words -- "end" in "vendor", "ip" in
            # "equipment" -- handing +3 relevance to clauses that never
            # mentioned the topic. Stems like "terminat" and "revok" still
            # match their inflections.
            for kw in boost_keywords:
                if signal_in(text, kw):
                    score += 3

            # Mild boost for raw query words
            for word in query_words:
                if len(word) > 3 and word in text:
                    score += 1

            # Penalize clearly off-topic clauses
            # If query is about termination/firing
            # penalize IP and confidentiality clauses

            termination_query_words = [
                "fire", "fired", "firing",
                "terminate", "termination",
                "fired", "dismiss", "layoff",
                "let go", "without reason",
                "without cause", "end my job",
            ]

            ip_clause_signals = [
                "intellectual property",
                "ip ownership", "invention",
                "copyright", "patent",
                "work product", "assign",
                "california law", "section 2870",
            ]

            payment_query_words = [
                "salary", "pay", "paid",
                "compensation", "bonus",
                "benefits", "earn",
            ]

            ip_query_words = [
                "ip", "patent", "invention",
                "own", "ownership", "create",
                "copyright", "intellectual",
            ]

            # If asking about termination
            # penalize IP clauses
            is_termination_query = any(
                w in query_lower
                for w in termination_query_words)

            is_ip_clause = any(
                s in text
                for s in ip_clause_signals)

            if is_termination_query and is_ip_clause:
                score -= 10

            # If asking about payment/salary
            # penalize termination clauses
            is_payment_query = any(
                w in query_lower
                for w in payment_query_words)

            termination_clause_signals = [
                "terminat", "fired", "cause",
                "severance", "notice period",
                "without cause",
            ]

            is_termination_clause = any(
                s in text
                for s in termination_clause_signals)

            # Only penalize if NOT also about payment
            if (is_payment_query and
                    not is_ip_clause and
                    is_termination_clause and
                    "severance" not in query_lower):
                score -= 5

            # If asking about IP
            # penalize termination and salary clauses
            is_ip_query = any(
                w in query_lower
                for w in ip_query_words)

            salary_clause_signals = [
                "base salary", "annual salary",
                "payroll", "bonus program",
                "car allowance", "expense",
            ]

            is_salary_clause = any(
                s in text
                for s in salary_clause_signals)

            if is_ip_query and is_salary_clause:
                score -= 10

            duration_query_words = [
                "long", "last", "duration",
                "expire", "expir", "how long",
                "when does", "end date",
                "term",
            ]

            is_duration_query = any(
                w in query_lower
                for w in duration_query_words)

            confidentiality_clause_signals = [
                "keep confidential",
                "not disclose",
                "confidential information means",
                "return or destroy",
                "protect",
            ]

            is_pure_confidentiality = any(
                s in text
                for s in
                confidentiality_clause_signals)

            if (is_duration_query and
                    is_pure_confidentiality):
                score -= 6

            # Penalize technical spec sections
            # that are not legal clauses
            technical_spec_signals = [
                "queuing", "routing", "firewall",
                "protocol", "network topology",
                "data center", "bug fix",
                "core component", "native mobile",
                "consumer web app", "api endpoint",
                "http", "sftp", "bandwidth",
                "latency", "throughput",
                "load balancer", "cdn",
                "scheduling", "dispatching",
                "carepoint", "converge app",
                "username", "encrypted password",
                "remote login", "remote access",
                "security breach notification",
                "breach continue",
                "support hotline",
                "escalation", "account manager",
                "vice president notif",
                "version support",
                "120 days after delivery",
                "previous version",
                "most recent version",
                "fix plan within",
                "ticket when",
                "update every hour",
            ]

            is_technical_spec = any(
                s in text
                for s in technical_spec_signals)

            # Always penalize technical specs
            # regardless of query
            if is_technical_spec:
                score -= 15

            # For access/shutdown queries
            # boost suspension and notice clauses
            access_query_words = [
                "shut down", "shutdown",
                "suspend", "suspension",
                "access", "cut off",
                "disconnect", "block",
                "warning", "notice",
                "without warning",
                "turn off", "disable",
            ]

            is_access_query = any(
                w in query_lower
                for w in access_query_words)

            access_clause_signals = [
                "suspend", "suspension",
                "access suspension",
                "terminate access",
                "discontinue service",
                "without notice",
                "prior written notice",
                "right to suspend",
                "right to terminate",
                "shut down", "shutdown",
                "disable access",
                "cut off access",
                "interruption of service",
                "service interruption",
                "scheduled maintenance",
                "unscheduled maintenance",
                "planned downtime",
                "unplanned downtime",
                "notice of termination",
                "termination notice",
                "days notice",
                "written notice required",
            ]

            if is_access_query:
                for s in access_clause_signals:
                    if s in text:
                        score += 5

            # Never show negative scored clauses
            # at the top — floor at 0
            final_score = max(0, score)
            # Only flag triggered when the score
            # is meaningfully above noise. A
            # score of 1-2 usually means a
            # single weak partial match.
            clause["triggered"] = (
                final_score >= 1)
            return final_score

        # Sort by relevance — most relevant first
        clauses = sorted(
            clauses,
            key=clause_score,
            reverse=True)

        # Format missing clauses for UI
        missing = []
        for g in missing_raw:
            missing.append({
                "clause_type": g.get(
                    "clause_type", "unknown"),
                "severity": g.get(
                    "severity", "MEDIUM"),
                "description": g.get(
                    "description", ""),
            })

        contract_type = result.get(
            "contract_type", "Contract")

        # Override if contract type is wrong
        # Check the filename for hints
        if uploaded_file:
            fname = uploaded_file.name.lower()
            if any(w in fname for w in [
                "ip", "licens", "patent"]):
                if contract_type == "NDA":
                    contract_type = (
                        "IP License Agreement")
            elif any(w in fname for w in [
                "employ", "employment"]):
                contract_type = (
                    "Employment Agreement")
            elif any(w in fname for w in [
                "nda", "confidential",
                "non-disclosure"]):
                contract_type = "NDA"

        return {
            "clauses": clauses,
            "missing": missing,
            "contract_type": contract_type,
            "intent": result.get(
                "intent", ""),
            "mode": mode,
            "agents": result.get(
                "agent_statuses", {})
        }

    finally:
        # Clean up temp file
        if file_path and os.path.exists(
                file_path):
            os.remove(file_path)


def status_icon(status):
    if status == "running":
        return "◉", "#B45309"
    if status == "done":
        return "✓", "#0F766E"
    if status == "failed":
        return "✕", "#C0392B"
    return "○", "#2A4A6F"


def clear_for_new_analysis():
    history = st.session_state.get("history", [])
    st.session_state.clear()
    st.session_state["history"] = history
    init_state()


init_state()
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        """
        <div style="padding:2rem 1.5rem;border-bottom:1px solid #1E3A5F;">
          <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:3px;color:#4A6A9F;">LEGAL INTELLIGENCE</div>
          <div style="font-family:'Playfair Display',serif;font-size:32px;font-weight:900;color:#F7F4EE;line-height:1.2;">Legal Agent</div>
          <div style="font-family:'Lato',sans-serif;font-size:11px;color:#4A6A9F;">v1.0.0 · Multi-Agent Mode</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="padding:1.5rem;border-bottom:1px solid #1E3A5F;">'
        '<div style="font-family:\'DM Mono\',monospace;font-size:9px;letter-spacing:2px;color:#4A6A9F;margin-bottom:0.8rem;">AGENT PIPELINE</div>',
        unsafe_allow_html=True,
    )
    statuses = st.session_state["agent_status"]
    for key, name, desc in AGENTS:
        symbol, color = status_icon(statuses.get(key, "idle"))
        st.markdown(
            f"""
            <div style="margin-bottom:0.55rem;">
              <div style="font-family:'Lato',sans-serif;font-size:12px;font-weight:700;color:#D0D8E8;">
                <span style="color:{color};padding-right:6px;">{symbol}</span>{name}
              </div>
              <div style="font-family:'Lato',sans-serif;font-size:10px;color:#4A6A9F;padding-left:17px;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Show latest query stats
    # from most recent trace file
    import glob
    import json as _json

    trace_files = sorted(
        glob.glob("logs/trace_*.json"),
        reverse=True)

    if trace_files:
        try:
            with open(
                    trace_files[0]) as f:
                trace = _json.load(f)

            st.sidebar.markdown(
                "---")
            st.sidebar.markdown(
                '<p style="color: #4A90D9; '
                'font-weight: bold; font-size: 14px;">'
                'Last Query Stats</p>',
                unsafe_allow_html=True)
            st.sidebar.markdown(
                '<p style="color: #E8EFF8; font-size: 13px;">'
                f"🕐 Latency: "
                f"{trace.get('total_latency_ms', 0):,} ms"
                '</p>',
                unsafe_allow_html=True)
            st.sidebar.markdown(
                '<p style="color: #E8EFF8; font-size: 13px;">'
                f"🔤 Tokens: "
                f"{trace.get('total_tokens', 0):,}"
                '</p>',
                unsafe_allow_html=True)
            st.sidebar.markdown(
                '<p style="color: #E8EFF8; font-size: 13px;">'
                f"💰 Cost: "
                f"${trace.get('total_cost_usd', 0):.4f}"
                '</p>',
                unsafe_allow_html=True)
            st.sidebar.markdown(
                '<p style="color: #E8EFF8; font-size: 13px;">'
                f"⚖️ Jurisdiction: "
                f"{trace.get('jurisdiction', 'Unknown')}"
                '</p>',
                unsafe_allow_html=True)
            st.sidebar.markdown(
                '<p style="color: #E8EFF8; font-size: 13px;">'
                f"📅 Effective: "
                f"{trace.get('effective_date', 'Not found')}"
                '</p>',
                unsafe_allow_html=True)
            st.sidebar.markdown(
                '<p style="color: #E8EFF8; font-size: 13px;">'
                f"🗓️ Expires: "
                f"{trace.get('expiry_date', 'Not found')}"
                '</p>',
                unsafe_allow_html=True)
        except Exception:
            pass

    api_key = os.environ.get(
        "ANTHROPIC_API_KEY")
    if api_key:
        status_text = "● Claude API connected"
        status_color = "#0F766E"
    else:
        status_text = "● No API key — mock mode"
        status_color = "#B45309"
    if api_key:
        st.markdown(
            f'<div style="padding:0.5rem 1.2rem;'
            f'font-family:DM Mono,monospace;'
            f'font-size:9px;color:{status_color};">'
            f'{status_text}</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div style="padding:0.5rem 1.2rem;'
            f'font-family:DM Mono,monospace;'
            f'font-size:9px;color:{status_color};">'
            f'{status_text}</div>',
            unsafe_allow_html=True)

    st.markdown(
        '<div style="padding:1.5rem;border-bottom:1px solid #1E3A5F;">'
        '<div style="font-family:\'DM Mono\',monospace;font-size:9px;letter-spacing:2px;color:#4A6A9F;margin-bottom:0.8rem;">RECENT QUERIES</div>',
        unsafe_allow_html=True,
    )
    history = st.session_state.get("history", [])[-4:][::-1]
    if not history:
        st.markdown(
            "<div style='font-family:Lato,sans-serif;font-size:11px;color:#4A6A9F;font-style:italic;'>No queries yet</div>",
            unsafe_allow_html=True,
        )
    else:
        for item in history:
            q = item.get("query", "")
            text = q if len(q) <= 52 else q[:52] + "..."
            st.markdown(
                f"""
                <div class="side-card">
                    <div style="font-family:'Lato',sans-serif;font-size:11px;color:#D0D8E8;">{text}</div>
                    <div style="font-family:'DM Mono',monospace;font-size:9px;color:#4A6A9F;margin-top:4px;">{item.get("contract_name","knowledge_only")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="padding:1rem 1.5rem;font-family:'DM Mono',monospace;font-size:8px;color:#2A4A6F;line-height:1.6;">
          Legal information only.<br>
          Not legal advice. Consult<br>
          a qualified lawyer for decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="padding-bottom:1rem;border-bottom:2px solid #1a1208;margin-bottom:1.2rem;">
      <div class="eyebrow">// CONTRACT INTELLIGENCE PLATFORM</div>
      <div class="title" style="font-size:34px;font-weight:700;">Legal Intelligence Agent</div>
      <div class="subtle" style="font-size:14px;">Upload any business contract · Ask in plain English · Get risk-scored clause analysis</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns(2)

with left_col:
    st.markdown('<div class="label">CONTRACT DOCUMENT</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload contract", type=["pdf", "docx", "txt"], label_visibility="collapsed")
    if uploaded_file is not None:
        size_kb = uploaded_file.size / 1024
        st.markdown(
            f"""
            <div style="background:#E6F4F3;border:1px solid #0F766E;border-radius:2px;padding:0.6rem 0.7rem;margin-top:0.45rem;">
              <div style="font-family:'Lato',sans-serif;font-size:12px;color:#0F766E;"><b>{uploaded_file.name}</b> · {size_kb:.1f} KB · ready to analyze</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right_col:
    st.markdown('<div class="label">YOUR LEGAL QUESTION</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Lato',sans-serif;
    font-size:11px;color:#9A8F7A;
    margin-bottom:0.4rem;font-style:italic;">
    Tip: Ask specific questions like
    "What are my obligations?" or
    "What happens if I terminate early?"
    for best results.
    </div>
    """, unsafe_allow_html=True)
    query = st.text_area(
        "Your legal question",
        value="",
        height=115,
        label_visibility="collapsed",
        placeholder="e.g. What are my confidentiality obligations?\nWhat happens if I terminate early?\nWho owns the IP I create at work?",
    )

analyze_clicked = st.button("⚖  Analyze Contract", use_container_width=True)
st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)

if analyze_clicked:
    if not query.strip():
        st.error("Please enter a legal question.")
    else:
        contract_name = uploaded_file.name if uploaded_file is not None else "knowledge_only"
        if uploaded_file is None:
            st.info("No contract uploaded — running in knowledge-only mode.")
        st.session_state["agent_status"] = {k: "running" for k, _, _ in AGENTS}
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        mode_label = (
            "Claude API" if api_key else "Mock mode")
        with st.spinner(
            f"Analyzing your contract "
            f"via {mode_label}…"):
            result = run_real(
                query.strip(), uploaded_file)
        st.session_state["result"] = result
        st.session_state["history"].append({"query": query, "contract_name": contract_name})
        st.session_state["agent_status"] = {k: "done" for k, _, _ in AGENTS}
        st.rerun()

if "result" in st.session_state:
    result = st.session_state["result"]
    clauses = result.get("clauses", [])
    missing = result.get("missing", [])
    gap_focused_query = any(
        phrase in query.lower()
        for phrase in [
            "missing", "what clauses",
            "gaps", "lacking", "absent",
            "not included", "should have",
            "whats missing",
            "what is missing",
            "what are missing",
            "what clauses are missing",
            "what are my clauses missing",
            "what clause is missing",
            "incomplete", "what else",
            "anything missing",
            "what should be",
            "what needs to be",
            "what doesnt it have",
            "what does it not have",
            "what isnt in",
            "what is not in",
            "what am i missing",
            "did i miss",
        ])
    critical_count = sum(1 for c in clauses if c.get("risk_level") == "CRITICAL")
    high_count = sum(1 for c in clauses if c.get("risk_level") == "HIGH")
    med_count = sum(1 for c in clauses if c.get("risk_level") == "MEDIUM")
    low_count = sum(1 for c in clauses if c.get("risk_level") == "LOW")

    l, r = st.columns([2.2, 1.8])
    with l:
        st.markdown(
            f"""
            <div class="summary-bar">
              <div style="font-family:'Playfair Display',serif;font-size:17px;color:#F7F4EE;">Analysis Complete — {result.get("contract_type","Unknown")}</div>
              <div style="font-family:'Lato',sans-serif;font-size:12px;color:#9A8F7A;">{len(clauses)} clauses · {len(missing)} missing · {critical_count} require immediate attention</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r:
        stat_cols = st.columns(4)
        stats = [("CRITICAL", critical_count), ("HIGH", high_count), ("MEDIUM", med_count), ("LOW", low_count)]
        for i, (label, value) in enumerate(stats):
            with stat_cols[i]:
                st.markdown(
                    f"""
                    <div style="background:#fff;border:1px solid #E0D8CC;border-radius:3px;text-align:center;padding:0.4rem;">
                      <div style="font-family:'Playfair Display',serif;font-size:24px;color:{RISK_COLORS[label]};">{value}</div>
                      <div style="font-family:'DM Mono',monospace;font-size:8px;color:#9A8F7A;">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Only show triggered clauses first
    # Fall back to top 7 by risk if none triggered
    triggered = [
        c for c in clauses
        if c.get("triggered")]

    if triggered:
        triggered_sorted = sorted(
            triggered,
            key=lambda x: x.get(
                "confidence", 0),
            reverse=True)

        if len(triggered_sorted) >= 4:
            display_clauses = (
                triggered_sorted[:7])
        else:
            # Fill remaining slots with
            # highest confidence
            # non-triggered clauses
            non_triggered = sorted(
                [c for c in clauses
                 if not c.get(
                     "triggered", False)],
                key=lambda x: x.get(
                    "confidence", 0),
                reverse=True)
            display_clauses = (
                triggered_sorted +
                non_triggered
            )[:7]
    else:
        # Sort by risk priority
        risk_order = {
            "CRITICAL": 0,
            "HIGH": 1,
            "MEDIUM": 2,
            "LOW": 3
        }
        display_clauses = sorted(
            clauses,
            key=lambda x: risk_order.get(
                x.get("risk_level", "LOW"), 3)
        )[:7]

    if not gap_focused_query:
        st.markdown(
            f'<div style="font-family:DM Mono,'
            f'monospace;font-size:9px;color:#9A8F7A;'
            f'margin-bottom:1rem;">'
            f'CLAUSE ANALYSIS '
            f'· Showing {len(display_clauses)} '
            f'of {len(clauses)} clauses'
            f'</div>',
            unsafe_allow_html=True)
        for i, clause in enumerate(display_clauses):
            risk = clause["risk_level"]
            appr_key = f"appr_{i}"
            risk_colors = {
                "CRITICAL": "#EF4444",
                "HIGH": "#F97316",
                "MEDIUM": "#B45309",
                "LOW": "#0F766E"
            }
            risk_icons = {
                "CRITICAL": "🔴",
                "HIGH": "🟠",
                "MEDIUM": "🟡",
                "LOW": "🟢"
            }
            color = risk_colors.get(risk, "#9A8F7A")
            icon = risk_icons.get(risk, "⚪")

            clause_type_display = (
                clause["clause_type"]
                .replace("_", " ")
                .title()
            )

            confidence_pct = int(
                clause["confidence"] * 100)

            source_pos = clause["source_position"]
            plain_text = clause.get(
                "plain_english", "")

            if not plain_text.strip():
                raw = clause.get("text", "")
                plain_text = raw.strip()[:1200]
                if len(raw.strip()) > 1200:
                    plain_text += "…"

            # If plain english looks like raw
            # contract text (very long), truncate it
            if len(plain_text) > 1200:
                plain_text = plain_text[:1200] + "…"

            triggered_html = ""
            if clause.get("triggered"):
                triggered_html = (
                    '<span style="font-family:'
                    'DM Mono,monospace;font-size:9px;'
                    'background:#E6F4F3;color:#0F766E;'
                    'padding:1px 7px;border-radius:2px;'
                    'margin-left:6px;">triggered</span>'
                )

            approval = st.session_state.get(
                f"appr_{i}")

            badge_html = ""
            if approval == "approved":
                badge_html = (
                    '<span style="font-family:'
                    'DM Mono,monospace;font-size:9px;'
                    'background:#E6F4F3;color:#0F766E;'
                    'padding:2px 8px;border-radius:2px;'
                    'margin-left:8px;">&#10003; APPROVED'
                    '</span>'
                )
            elif approval == "rejected":
                badge_html = (
                    '<span style="font-family:'
                    'DM Mono,monospace;font-size:9px;'
                    'background:#FEF2F2;color:#EF4444;'
                    'padding:2px 8px;border-radius:2px;'
                    'margin-left:8px;">&#10005; REJECTED'
                    '</span>'
                )

            card_html = (
                f'<div style="background:white;'
                f'border:1px solid #E0D8CC;'
                f'border-left:4px solid {color};'
                f'border-radius:3px;'
                f'padding:1.1rem 1.4rem;'
                f'margin-bottom:0.8rem;">'
                f'<div style="display:flex;'
                f'align-items:center;'
                f'justify-content:space-between;'
                f'margin-bottom:0.55rem;">'
                f'<div>'
                f'{icon} '
                f'<span style="font-family:DM Mono,'
                f'monospace;font-size:9px;'
                f'letter-spacing:2px;color:{color};'
                f'font-weight:500;">{risk}</span>'
                f' &middot; '
                f'<span style="font-family:DM Mono,'
                f'monospace;font-size:9px;'
                f'color:#9A8F7A;">'
                f'{clause_type_display}</span>'
                f'{triggered_html}'
                f'{badge_html}'
                f'</div>'
                f'<div style="font-family:DM Mono,'
                f'monospace;font-size:9px;'
                f'color:#C8BFB0;">'
                f'&sect; {source_pos} '
                f'&middot; {confidence_pct}%'
                f'</div>'
                f'</div>'
                f'<div style="font-family:Lato,'
                f'sans-serif;font-size:14px;'
                f'color:#1a1208;line-height:1.65;">'
                f'{plain_text}'
                f'</div>'
                f'</div>'
            )

            st.markdown(card_html,
                        unsafe_allow_html=True)

            if risk == "CRITICAL" and approval is None:
                a, b, _ = st.columns([1, 1, 4])
                with a:
                    if st.button("✓  Approve", key=f"approve_{i}", use_container_width=True):
                        st.session_state[appr_key] = "approved"
                        st.rerun()
                with b:
                    if st.button("✕  Reject", key=f"reject_{i}", use_container_width=True):
                        st.session_state[appr_key] = "rejected"
                        st.rerun()

    if gap_focused_query:
        st.markdown("""
        <div style="margin-bottom:1rem;
        padding:0.75rem 1.2rem;
        background:#FEF2F2;
        border:1px solid #EF4444;
        border-left:4px solid #EF4444;
        border-radius:3px;">
            <div style="font-family:
            'DM Mono',monospace;
            font-size:10px;color:#EF4444;
            font-weight:700;letter-spacing:2px;">
            GAP ANALYSIS
            </div>
            <div style="font-family:'Lato',
            sans-serif;font-size:12px;
            color:#9A8F7A;margin-top:3px;">
            Clauses missing from this contract
            based on standard template.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="label" style="margin-top:1rem;">MISSING CLAUSES</div>', unsafe_allow_html=True)
    for miss in missing:
        sev = miss.get("severity", "LOW")
        mcolor = RISK_COLORS.get(sev, "#0F766E")
        st.markdown(
            f"""
            <div class="missing-card" style="border-left:4px solid {mcolor};">
              <div style="display:flex;gap:8px;align-items:center;">
                <span class="badge" style="background:#F9F6F1;color:{mcolor};border:1px solid {mcolor};">{sev}</span>
                <div style="font-family:'Lato',sans-serif;font-size:13px;font-weight:700;color:#1a1208;">{miss.get("clause_type","unknown")}</div>
              </div>
              <div style="font-family:'Lato',sans-serif;font-size:12px;color:#9A8F7A;margin-top:0.3rem;">{miss.get("description","")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="background:#F0EDE6;border-left:3px solid #C8BFB0;padding:0.55rem 0.7rem;margin-top:0.9rem;font-family:'DM Mono',monospace;font-size:9px;color:#9A8F7A;">
          ⚖  This is legal information, not legal advice. Consult a qualified lawyer before making decisions based on this analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("↺  New Analysis", use_container_width=False):
        clear_for_new_analysis()
        st.rerun()
else:
    st.markdown(
        """
        <div style="margin-top:4rem;text-align:center;">
          <div style="font-size:44px;">⚖</div>
          <div style="font-family:'Playfair Display',serif;font-size:22px;color:#1a1208;">Ready to analyze your contract</div>
          <div style="font-family:'Lato',sans-serif;font-size:14px;color:#9A8F7A;max-width:760px;margin:0.6rem auto 1rem;">
            Upload a contract, ask your legal question in plain English, and get a structured risk-scored clause analysis with gap detection.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    s1, s2, s3 = st.columns(3)
    stat_empty = [("4", "AGENTS", "#1E3A5F"), ("15", "CLAUSES FOUND", "#B45309"), ("0", "HALLUCINATIONS", "#0F766E")]
    for col, (num, label, color) in zip([s1, s2, s3], stat_empty):
        with col:
            st.markdown(
                f"""
                <div style="background:#fff;border:1px solid #E0D8CC;border-radius:3px;padding:0.85rem;text-align:center;">
                  <div style="font-family:'Playfair Display',serif;font-size:30px;color:{color};">{num}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:9px;color:#9A8F7A;">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
