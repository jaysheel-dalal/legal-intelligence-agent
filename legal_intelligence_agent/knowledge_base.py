from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class KnowledgeBaseError(Exception):
    pass


_SEVERITY_LEVELS = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}


def _normalize_key(s: str) -> str:
    return (s or "").strip().lower()


class KnowledgeBase:
    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        if not self.knowledge_dir.exists():
            raise KnowledgeBaseError(f"knowledge_dir does not exist: {self.knowledge_dir}")

        self._frameworks_dir = self.knowledge_dir / "legal_frameworks"
        self._templates_dir = self.knowledge_dir / "contract_templates"

        if not self._frameworks_dir.exists() or not self._templates_dir.exists():
            raise KnowledgeBaseError("knowledge_dir is missing required subdirectories")

        self._frameworks: Dict[str, Dict[str, Any]] = {}
        self._templates: Dict[str, Dict[str, Any]] = {}

        self._load_json_files()
        self._validate_loaded_data()

    def _load_json_files(self) -> None:
        # Load all JSON; fail fast on malformed JSON.
        for p in sorted(self._frameworks_dir.glob("*.json")):
            data = self._load_single_json(p)
            framework_name = _normalize_key(str(data.get("framework", "")))
            if not framework_name:
                raise KnowledgeBaseError(f"framework file missing 'framework': {p.name}")
            self._frameworks[framework_name] = data

        for p in sorted(self._templates_dir.glob("*.json")):
            data = self._load_single_json(p)
            contract_type = _normalize_key(str(data.get("contract_type", "")))
            if not contract_type:
                raise KnowledgeBaseError(f"template file missing 'contract_type': {p.name}")
            self._templates[contract_type] = data

    def _load_single_json(self, path: Path) -> Dict[str, Any]:
        try:
            raw = path.read_text(encoding="utf-8")
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise KnowledgeBaseError(f"malformed json: {path.name}") from e
        except Exception as e:  # noqa: BLE001
            raise KnowledgeBaseError(f"failed to load json: {path.name}") from e

    def _validate_loaded_data(self) -> None:
        # Frameworks schema checks (light validation, no external schema libs).
        for name, fw in self._frameworks.items():
            required_framework_keys = {"framework", "description", "applies_to", "obligations"}
            if not required_framework_keys.issubset(set(fw.keys())):
                raise KnowledgeBaseError(f"framework missing keys: {name}")
            obligations = fw.get("obligations", [])
            if not isinstance(obligations, list) or not obligations:
                raise KnowledgeBaseError(f"framework obligations missing/empty: {name}")
            for ob in obligations:
                if not _SEVERITY_LEVELS.issuperset({str(ob.get("severity", "")).upper()}):
                    raise KnowledgeBaseError(f"invalid severity in framework: {name}")
                if "clause_types_affected" not in ob:
                    raise KnowledgeBaseError(f"framework obligation missing clause_types_affected: {name}")

        for name, tpl in self._templates.items():
            if "contract_type" not in tpl or "required_clauses" not in tpl:
                raise KnowledgeBaseError(f"template missing keys: {name}")

    def list_contract_types(self) -> List[str]:
        return sorted([tpl.get("contract_type", "") for tpl in self._templates.values()], key=_normalize_key)

    def list_frameworks(self) -> List[str]:
        return sorted([fw.get("framework", "") for fw in self._frameworks.values()], key=_normalize_key)

    def _load_template(self, template_name: str) -> Dict[str, Any]:
        tkey = _normalize_key(template_name)
        for tpl in self._templates.values():
            ct = _normalize_key(str(tpl.get("contract_type", "")))
            if ct == tkey:
                return tpl
        raise KnowledgeBaseError(f"unknown template: {template_name}")

    def _get_template(self,
                      contract_type: str) -> Dict[str, Any]:

        type_map = {
            "nda": "nda",
            "non-disclosure": "nda",
            "confidentiality": "nda",
            "ip license": "ip_license",
            "ip licensing": "ip_license",
            "license agreement": "ip_license",
            "licensing agreement": "ip_license",
            "patent license": "ip_license",
            "employment": "employment",
            "employment agreement": "employment",
            "offer letter": "employment",
            "saas": "saas",
            "software": "saas",
            "subscription": "saas",
            "service agreement": "saas",
            "vendor": "vendor",
            "supply": "vendor",
            "purchase": "vendor",
            "supplier": "vendor",
            "partnership": "partnership",
            "joint venture": "partnership",
            "general partner": "partnership",
            "limited partner": "partnership",
            "partnership agreement": "partnership",
        }

        # Normalize contract type
        ct_lower = contract_type.lower().strip()
        ct_normalized = ct_lower.replace(
            "_", " ").replace("-", " ")

        # Find matching template
        template_name = "nda"  # default
        for key, val in type_map.items():
            if (key in ct_lower or
                    key in ct_normalized):
                template_name = val
                break

        aliases = {
            "nda": ["nda", "non-disclosure agreement"],
            "ip_license": ["ip license agreement"],
            "employment": ["employment agreement"],
            "saas": ["saas agreement"],
            "vendor": ["vendor"],
        }
        for candidate in aliases.get(template_name, [template_name]):
            try:
                return self._load_template(candidate)
            except KnowledgeBaseError:
                continue

        raise KnowledgeBaseError(f"unknown contract_type: {contract_type}")

    def get_required_clauses(self, contract_type: str) -> List[Dict[str, Any]]:
        tpl = self._get_template(contract_type)
        required = tpl.get("required_clauses", [])
        # Return a list (each entry is a clause descriptor).
        return list(required)

    def get_framework(self, framework_name: str) -> Dict[str, Any]:
        key = _normalize_key(framework_name)
        fw = self._frameworks.get(key)
        if fw is None:
            raise KnowledgeBaseError(f"unknown framework: {framework_name}")
        return fw

    def get_clause_standard(self, clause_type: str) -> Dict[str, Any]:
        ckey = _normalize_key(clause_type)

        found_in: List[str] = []
        first_match: Optional[Dict[str, Any]] = None

        for tpl in self._templates.values():
            contract_type = _normalize_key(str(tpl.get("contract_type", "")))
            required = tpl.get("required_clauses", []) or []
            optional = tpl.get("optional_clauses", []) or []
            all_clauses = list(required) + list(optional)
            for clause in all_clauses:
                if _normalize_key(str(clause.get("clause_type", ""))) == ckey:
                    found_in.append(tpl.get("contract_type", ""))
                    if first_match is None:
                        first_match = clause

        if first_match is None:
            raise KnowledgeBaseError(f"unknown clause_type: {clause_type}")

        return {
            "clause_type": first_match.get("clause_type", clause_type),
            "title": first_match.get("title"),
            "description": first_match.get("description"),
            "severity_if_missing": first_match.get("severity_if_missing"),
            "found_in": sorted(list(dict.fromkeys(found_in)), key=_normalize_key),
        }

    def check_gaps(self, found_clause_types: List[str], contract_type: str) -> List[Dict[str, Any]]:
        tpl_required = self.get_required_clauses(contract_type)
        required_types = {_normalize_key(c.get("clause_type", "")) for c in tpl_required}

        found_types = {_normalize_key(x) for x in (found_clause_types or []) if _normalize_key(x)}
        missing_types = sorted(list(required_types - found_types), key=_normalize_key)

        missing: List[Dict[str, Any]] = []
        for c in tpl_required:
            sev = c.get("severity_if_missing", c.get("severity"))
            if _normalize_key(str(c.get("clause_type", ""))) in missing_types:
                missing.append(
                    {
                        "clause_type": c.get("clause_type"),
                        "severity_if_missing": sev,
                        "title": c.get("title"),
                        "description": c.get("description"),
                    }
                )

        # Validate severity domain.
        for m in missing:
            sev = str(m.get("severity_if_missing", "")).upper()
            if sev not in _SEVERITY_LEVELS:
                raise KnowledgeBaseError(f"invalid severity_if_missing: {sev}")
            m["severity_if_missing"] = sev
        return missing

