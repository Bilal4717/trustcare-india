"""Keyword and rule-based logic used when no OpenAI (or other) LLM is configured."""

import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple


def detect_intent_heuristic(user_text: str) -> str:
    t = (user_text or "").lower()
    if not t.strip():
        return "out_of_scope"
    oos = ["weather", "stock", "bitcoin", "recipe", "python tutorial", "who won"]
    if any(x in t for x in oos):
        return "out_of_scope"
    if any(k in t for k in ("desert", "gap", "underserved", "lack of", "no hospital", "shortage")):
        return "desert"
    if any(k in t for k in ("audit", "verify", "really have", "does it have", "claim")):
        return "audit"
    if any(k in t for k in ("trust", "reliable", "fake", "contradict", "truth")):
        return "trust"
    if any(
        k in t
        for k in (
            "find",
            "search",
            "where",
            "nearest",
            "list",
            "which",
            "hospital",
            "clinic",
            "facility",
            "icu",
            "nicu",
            "dialysis",
            "oncology",
            "appendectomy",
            "surgery",
            "emergency",
            "pin",
            "bihar",
            "rural",
        )
    ):
        return "query"
    return "query"


def keyword_capability_flags(bag: str) -> SimpleNamespace:
    """Infer booleans from concatenated facility text (same roles as FacilityCapabilities)."""
    b = (bag or "").lower()
    return SimpleNamespace(
        has_icu=any(x in b for x in ("icu", "intensive care", "critical care", "ventilator")),
        has_emergency=any(x in b for x in ("emergency", "trauma", "24/7", "24x7", "casualty")),
        has_surgery=any(x in b for x in ("surgery", "operation theatre", "operation theater", " ot", "(ot)", "ot ")),
        has_dialysis=any(x in b for x in ("dialysis", "nephrolog", "hemodialysis")),
        has_oncology=any(x in b for x in ("oncology", "oncologist", "chemotherapy", "radiation")),
        has_neonatal=any(x in b for x in ("neonatal", "nicu", "neonatolog", "incubator")),
        num_doctors=_first_int_near(b, ["doctor", "physician", "staff"]),
        bed_capacity=_first_int_near(b, ["bed", "capacity"]),
        operates_24_7=any(x in b for x in ("24/7", "24x7", "round the clock", "24 hour")),
        confidence_note="Heuristic keyword scan (no generative LLM).",
    )


def _first_int_near(text: str, keywords: List[str]) -> Optional[int]:
    for kw in keywords:
        m = re.search(rf"{re.escape(kw)}[^\d]{{0,20}}(\d{{1,5}})", text, re.I)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None


def validator_rule_based(state: Dict[str, Any]) -> Tuple[bool, str]:
    intent = state.get("intent") or ""
    if intent == "out_of_scope":
        return True, "VALID: out-of-scope response"
    if intent == "query":
        if not state.get("retrieved_docs"):
            return False, "INVALID: no facilities retrieved for query intent"
        return True, "VALID: query has retrieved rows"
    if intent == "audit":
        ar = state.get("audit_result") or {}
        if ar.get("status") == "not_found":
            return False, "INVALID: audit target not found in index"
        return True, "VALID: audit produced structured result"
    if intent == "trust":
        if state.get("trust_score") is None:
            return False, "INVALID: missing trust score"
        return True, "VALID: trust score computed"
    if intent == "desert":
        if state.get("desert_regions") is None:
            return False, "INVALID: desert analysis missing"
        return True, "VALID: desert regions present"
    return True, "VALID: default"


def format_validator_verdict(valid: bool, reason: str) -> str:
    return ("VALID: " if valid else "INVALID: ") + reason
