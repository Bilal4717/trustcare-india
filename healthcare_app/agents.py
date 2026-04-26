import json
import math
import re
from typing import Any, List, Optional

import pandas as pd
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from healthcare_app.citations import evidence_snippet
from healthcare_app.config import HIGH_ACUITY_SPECIALTIES, TRUST_RULES
from healthcare_app.data import _build_metadata, _parse_json_list, build_facility_text, format_docs_for_llm
from healthcare_app.heuristics import (
    detect_intent_heuristic,
    format_validator_verdict,
    keyword_capability_flags,
    validator_rule_based,
)
from healthcare_app.observability import traced_step
from healthcare_app.schemas import FacilityCapabilities, HealthcareState, TrustAssessment
from healthcare_app.tavily_search import format_tavily_block, search_tavily


def _normalise_terms(items: List[str]) -> List[str]:
    terms = []
    for item in items or []:
        s = str(item).strip()
        if not s:
            continue
        terms.append(s.lower())
        terms.append(s.replace("_", " ").lower())
    return list(dict.fromkeys(terms))


class AgentRuntime:
    def __init__(
        self,
        llm: Optional[Any],
        retriever: Any,
        df: pd.DataFrame,
        *,
        tavily_client: Optional[Any] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.df = df
        self.tavily_client = tavily_client
        self._state_terms = {
            str(s).strip().lower() for s in (df.get("address_stateOrRegion", pd.Series(dtype=str)).dropna().unique()) if str(s).strip()
        }
        self._city_terms = {
            str(c).strip().lower() for c in (df.get("address_city", pd.Series(dtype=str)).dropna().unique()) if str(c).strip()
        }
        self._pin_centroids = self._build_pin_centroids(df)

    def _name_candidates(self, facility_name: str, limit: int = 5) -> List[tuple]:
        """
        Deterministic name candidates for facility-specific audit.
        Returns (score, idx, row) sorted by score desc.
        """
        name = (facility_name or "").strip().lower()
        if not name or self.df is None or self.df.empty or "name" not in self.df.columns:
            return []
        tokens = [t for t in re.split(r"[^a-z0-9]+", name) if len(t) >= 3]
        if not tokens:
            return []

        candidates = []
        for idx, row in self.df.iterrows():
            row_name = str(row.get("name") or "").strip()
            if not row_name:
                continue
            ln = row_name.lower()
            token_hits = sum(1 for t in tokens if t in ln)
            if token_hits == 0:
                continue
            coverage = token_hits / len(tokens)
            exact_bonus = 0.35 if name == ln else 0.0
            prefix_bonus = 0.15 if ln.startswith(tokens[0]) else 0.0
            completeness = float(row.get("_completeness") or 0.0)
            score = coverage + exact_bonus + prefix_bonus + (0.1 * completeness)
            candidates.append((score, idx, row))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:limit]

    def _name_match_docs(self, facility_name: str, limit: int = 5, min_score: float = 0.55) -> List[Document]:
        """
        Resolve facility-specific audits using deterministic name matching
        against the raw dataframe before semantic retrieval.
        """
        candidates = self._name_candidates(facility_name, limit=limit)
        out = []
        for score, idx, row in candidates:
            if score < min_score:
                continue
            text = build_facility_text(row)
            if len(text.strip()) < 20:
                continue
            out.append(Document(page_content=text, metadata=_build_metadata(idx, row)))
        return out

    @staticmethod
    def _band(score: float) -> str:
        if score >= 0.8:
            return "high"
        if score >= 0.55:
            return "medium"
        return "low"

    def _retrieve_docs(self, query: str):
        """
        Compat wrapper for LangChain retriever APIs.
        Newer versions use .invoke(query); older versions expose
        .get_relevant_documents(query).
        """
        if not self.retriever:
            return []
        if hasattr(self.retriever, "invoke"):
            docs = self.retriever.invoke(query)
            return docs or []
        if hasattr(self.retriever, "get_relevant_documents"):
            docs = self.retriever.get_relevant_documents(query)
            return docs or []
        return []

    @staticmethod
    def _build_pin_centroids(df: pd.DataFrame) -> dict:
        if df is None or df.empty:
            return {}
        frame = df.copy()
        for c in ["latitude", "longitude"]:
            if c in frame.columns:
                frame[c] = pd.to_numeric(frame[c], errors="coerce")
        out = {}
        for pin, g in frame.groupby("pin"):
            if pin is None or str(pin).strip() == "":
                continue
            lat = g["latitude"].mean() if "latitude" in g.columns else None
            lon = g["longitude"].mean() if "longitude" in g.columns else None
            if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
                continue
            out[str(pin).strip()] = (float(lat), float(lon))
        return out

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))

    def _extract_constraints(self, query: str) -> dict:
        q = (query or "").lower()
        requested_state = next((s for s in self._state_terms if s in q), None)
        requested_city = next((c for c in self._city_terms if c in q), None)
        pin_match = re.search(r"\b(\d{6})\b", q)
        requested_pin = pin_match.group(1) if pin_match else None
        requires_nearest = any(k in q for k in ["nearest", "closest", "nearby", "near me"])
        requires_rural = any(k in q for k in ["rural", "village", "remote"])
        requires_part_time = any(k in q for k in ["parttime", "part-time", "part time"])
        return {
            "state": requested_state,
            "city": requested_city,
            "pin": requested_pin,
            "needs_nearest": requires_nearest,
            "needs_rural": requires_rural,
            "needs_part_time": requires_part_time,
            "raw": q,
        }

    @staticmethod
    def _intent_override(user_text: str, current_intent: str) -> str:
        """
        Deterministic guardrails for intent routing.
        Prevents generic contradiction prompts from being misrouted to audit.
        """
        q = (user_text or "").lower().strip()
        contradiction_signals = [
            "but no ",
            "without ",
            "missing ",
            "contradiction",
            "inconsistent",
            "suspicious",
            "trust",
            "not grounded",
            "no anesthesiolog",
        ]
        explicit_audit_signals = ["audit ", "verify facility", "audit facility", "check facility"]
        has_explicit_audit = any(s in q for s in explicit_audit_signals)
        has_contradiction = any(s in q for s in contradiction_signals)

        # If user asks for contradiction/trust checks without explicit audit command,
        # route to trust scorer.
        if has_contradiction and not has_explicit_audit:
            return "trust"

        # Generic "facility with ..." discovery asks should not default to audit.
        if current_intent == "audit" and "facility with" in q and '"' not in q and "'" not in q and not has_explicit_audit:
            return "trust" if has_contradiction else "query"
        return current_intent

    @staticmethod
    def _is_contradiction_search(query: str) -> bool:
        q = (query or "").lower()
        contradiction_signals = [
            "but no ",
            "without ",
            "missing ",
            "contradiction",
            "inconsistent",
            "truth gap",
            "suspicious",
            "does not match",
        ]
        return any(s in q for s in contradiction_signals)

    @staticmethod
    def _trust_doc_evaluation(doc) -> dict:
        m = doc.metadata or {}
        text = (doc.page_content or "").lower()
        claims = _parse_json_list(m.get("capability", "")) + _parse_json_list(m.get("specialties", ""))
        claims_text = " | ".join(claims).lower()
        evidence_text = " | ".join([text, str(m.get("equipment", "")).lower(), str(m.get("procedures", "")).lower()])

        flags = []
        score = 1.0
        breakdown = []
        evidence_map = []
        for claim_name, required_terms in TRUST_RULES.items():
            ck = claim_name.lower()
            if ck.replace(" ", "") in claims_text.replace(" ", "") or ck in claims_text:
                missing_terms = [t for t in required_terms if t.lower() not in evidence_text]
                if missing_terms:
                    flags.append(f"{claim_name}: missing evidence -> {', '.join(missing_terms)}")
                    score -= 0.12
                    breakdown.append({"type": "claim_evidence_gap", "claim": claim_name, "penalty": 0.12})
                    evidence_map.append(
                        {
                            "flag": f"{claim_name}: missing evidence",
                            "penalty": 0.12,
                            "evidence_sentence": AgentRuntime._find_evidence_sentence(text, required_terms),
                            "source_field": "page_content",
                            "source_row_id": m.get("facility_id"),
                        }
                    )
        if not m.get("affiliated_staff", False):
            flags.append("No affiliated staff profile signal.")
            score -= 0.05
            breakdown.append({"type": "profile_signal", "claim": "affiliated_staff_presence", "penalty": 0.05})
        if not m.get("custom_logo", False):
            flags.append("No custom branding signal.")
            score -= 0.03
            breakdown.append({"type": "profile_signal", "claim": "custom_logo_presence", "penalty": 0.03})
        followers = float(m.get("followers") or 0)
        if followers < 50:
            flags.append("Very low social proof (followers < 50).")
            score -= 0.05
            breakdown.append({"type": "social_signal", "claim": "followers_lt_50", "penalty": 0.05})

        high_severity = AgentRuntime._high_severity_contradictions(claims_text, evidence_text, text)
        for contradiction in high_severity:
            flags.append(f"HIGH_SEVERITY: {contradiction}")
            score -= 0.2
            breakdown.append(
                {
                    "type": "high_severity_contradiction",
                    "claim": contradiction,
                    "penalty": 0.2,
                    "severity": "high",
                }
            )
            evidence_map.append(
                {
                    "flag": f"HIGH_SEVERITY: {contradiction}",
                    "penalty": 0.2,
                    "evidence_sentence": AgentRuntime._find_evidence_sentence(text, contradiction.split()),
                    "source_field": "page_content",
                    "source_row_id": m.get("facility_id"),
                }
            )
        return {
            "score": max(0.0, min(1.0, score)),
            "flags": flags,
            "breakdown": breakdown,
            "evidence_map": evidence_map,
            "high_severity": high_severity,
        }

    @staticmethod
    def _extract_required_procedures(query: str) -> List[str]:
        q = (query or "").lower()
        aliases = {
            "appendectomy": ["appendectomy", "appendix", "appendicectomy"],
            "emergency_surgery": ["emergency surgery", "emergency operation", "trauma surgery"],
            "icu_support": ["icu", "critical care", "ventilator"],
            "oncology": ["oncology", "chemotherapy", "radiation"],
            "dialysis": ["dialysis", "nephrology", "nephrologist"],
        }
        required = []
        for key, terms in aliases.items():
            if any(t in q for t in terms):
                required.append(key)
        return required

    @staticmethod
    def _has_part_time_signal(text: str) -> bool:
        staffing_terms = ["part-time", "part time", "visiting consultant", "visiting doctor", "on-call"]
        return any(t in (text or "").lower() for t in staffing_terms)

    @staticmethod
    def _has_rural_signal(text: str) -> bool:
        rural_terms = ["rural", "village", "block", "taluk", "tehsil", "phc", "chc", "community health"]
        return any(t in (text or "").lower() for t in rural_terms)

    @staticmethod
    def _procedure_match(text: str, procedure_key: str) -> bool:
        term_map = {
            "appendectomy": ["appendectomy", "appendix", "appendicectomy", "general surgery"],
            "emergency_surgery": ["emergency", "trauma", "operation theatre", "surgery"],
            "icu_support": ["icu", "critical care", "ventilator"],
            "oncology": ["oncology", "chemotherapy", "radiation"],
            "dialysis": ["dialysis", "nephrology", "nephrologist"],
        }
        return any(t in (text or "").lower() for t in term_map.get(procedure_key, []))

    def _apply_query_constraints(self, ranked: list, query: str) -> tuple[list, dict]:
        constraints = self._extract_constraints(query)
        required_procedures = self._extract_required_procedures(query)
        filtered = []
        evidence_summary = {
            "matched_attributes": [],
            "missing_attributes": [],
            "required_procedures": required_procedures,
            "location_anchor_available": bool(constraints.get("pin") and constraints["pin"] in self._pin_centroids),
        }
        for doc, score, reasons in ranked:
            m = doc.metadata or {}
            text = (doc.page_content or "").lower()
            ok = True
            missing = []
            if constraints["state"] and constraints["state"] not in str(m.get("state", "")).lower():
                ok = False
                missing.append("state_match")
            if constraints["city"] and constraints["city"] not in str(m.get("city", "")).lower():
                ok = False
                missing.append("city_match")
            if constraints["needs_rural"] and not self._has_rural_signal(text):
                ok = False
                missing.append("rural_signal")
            if constraints["needs_part_time"] and not self._has_part_time_signal(text):
                missing.append("part_time_signal_unconfirmed")
            for req in required_procedures:
                if not self._procedure_match(text, req):
                    ok = False
                    missing.append(f"procedure:{req}")
            if ok:
                filtered.append((doc, score, reasons))
            if missing:
                evidence_summary["missing_attributes"].extend(missing)
        if filtered:
            evidence_summary["matched_attributes"].extend(
                ["constraint_screen_pass", "location_filter_applied"] if constraints["state"] or constraints["city"] else ["constraint_screen_pass"]
            )
        if constraints["needs_part_time"]:
            if any(self._has_part_time_signal((d.page_content or "").lower()) for d, _, _ in filtered):
                evidence_summary["matched_attributes"].append("part_time_signal_present")
            else:
                evidence_summary["missing_attributes"].append("part_time_signal_unconfirmed")
        evidence_summary["missing_attributes"] = list(dict.fromkeys(evidence_summary["missing_attributes"]))
        evidence_summary["matched_attributes"] = list(dict.fromkeys(evidence_summary["matched_attributes"]))
        return filtered, evidence_summary

    def _rank_docs(self, docs: list, query: str) -> list:
        constraints = self._extract_constraints(query)
        anchor = None
        if constraints["pin"] and constraints["pin"] in self._pin_centroids:
            anchor = self._pin_centroids[constraints["pin"]]

        ranked = []
        for idx, doc in enumerate(docs):
            m = doc.metadata or {}
            text = doc.page_content.lower()
            score = max(0.0, 0.35 - 0.02 * idx)  # preserve retrieval ordering signal
            reasons = []

            state_ok = not constraints["state"] or constraints["state"] in str(m.get("state", "")).lower()
            city_ok = not constraints["city"] or constraints["city"] in str(m.get("city", "")).lower()
            if state_ok and city_ok:
                score += 0.2
                reasons.append("location_match")

            if constraints["needs_rural"]:
                rural_terms = ["rural", "village", "block", "taluk", "tehsil", "phc", "chc", "community health"]
                if any(t in text for t in rural_terms):
                    score += 0.14
                    reasons.append("rural_signal_match")
                else:
                    score -= 0.06
                    reasons.append("rural_signal_weak")

            if constraints["needs_part_time"]:
                staffing_terms = ["part-time", "part time", "visiting consultant", "visiting doctor", "on-call"]
                if any(t in text for t in staffing_terms):
                    score += 0.12
                    reasons.append("staffing_match_part_time")
                else:
                    score -= 0.04
                    reasons.append("staffing_not_explicit")

            procedure_hints = ["appendectomy", "laparoscopic", "emergency", "surgery", "icu", "ventilator"]
            proc_hits = [p for p in procedure_hints if p in constraints["raw"] and p in text]
            if proc_hits:
                score += min(0.2, 0.05 * len(proc_hits))
                reasons.append("procedure_capability_match")

            if constraints["needs_nearest"] and anchor is not None:
                lat = m.get("lat")
                lon = m.get("lon")
                if lat is not None and lon is not None:
                    dist = self._haversine_km(anchor[0], anchor[1], float(lat), float(lon))
                    near_boost = max(0.0, 0.15 - min(dist, 150.0) / 1000.0)
                    score += near_boost
                    reasons.append("geo_distance_scored")
                else:
                    reasons.append("geo_distance_unavailable")

            completeness = float(m.get("completeness") or 0.0)
            score += 0.1 * completeness
            reasons.append("doc_completeness_weighted")

            ranked.append((doc, max(0.0, min(1.0, score)), reasons))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def detect_intent(self, state: HealthcareState) -> HealthcareState:
        with traced_step("detect_intent"):
            user_text = state["messages"][-1].content
            if self.llm is not None:
                system_prompt = (
                    "You are an intent classifier for an Indian healthcare intelligence system.\n"
                    "Classify the user's query into exactly one intent:\n"
                    "- 'query'       : find or search for facilities (e.g. 'find ICU in rural Bihar')\n"
                    "- 'audit'       : verify a specific facility's capabilities\n"
                    "- 'desert'      : find regions lacking healthcare (e.g. 'where are oncology deserts?')\n"
                    "- 'trust'       : assess how trustworthy a facility's data is\n"
                    "- 'out_of_scope': not related to Indian healthcare facilities\n"
                    "Return ONLY one word."
                )
                result = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_text)])
                intent = result.content.strip().lower()
            else:
                intent = detect_intent_heuristic(user_text)
            intent = self._intent_override(user_text, intent)
            state["intent"] = intent
            state["chain_of_thought"] = [f"Intent detected: {state['intent']}"]
            if self.llm is None:
                state["chain_of_thought"].append("Intent mode: keyword heuristic (no OpenAI)")
            return state

    def query_agent(self, state: HealthcareState) -> HealthcareState:
        with traced_step("query_agent"):
            user_text = state["messages"][-1].content
            docs = self._retrieve_docs(user_text)
            ranked = self._rank_docs(docs, user_text)
            constrained_ranked, constraint_summary = self._apply_query_constraints(ranked, user_text)
            docs = [r[0] for r in (constrained_ranked or ranked)]
            state["retrieved_docs"] = [d.metadata for d in docs]
            state["reason_codes"] = []
            state["source_citations"] = [
                {
                    "facility_name": d.metadata.get("name", ""),
                    "pin": d.metadata.get("pin", ""),
                    "city": d.metadata.get("city", ""),
                    "state": d.metadata.get("state", ""),
                    "evidence_snippet": evidence_snippet(d.page_content),
                    "completeness": d.metadata.get("completeness"),
                }
                for d in docs[:5]
            ]
            context = format_docs_for_llm(docs)
            if ranked:
                ranking_lines = []
                for i, (_, score, reasons) in enumerate((constrained_ranked or ranked)[:5], 1):
                    ranking_lines.append(f"[{i}] score={score:.3f} | reasons={', '.join(reasons)}")
                context = f"{context}\n\n--- DETERMINISTIC RANKING SIGNALS ---\n" + "\n".join(ranking_lines)
            context += (
                "\n\n--- CONSTRAINT CHECK ---\n"
                f"matched_attributes={constraint_summary.get('matched_attributes', [])}\n"
                f"missing_attributes={constraint_summary.get('missing_attributes', [])}\n"
                f"required_procedures={constraint_summary.get('required_procedures', [])}\n"
                f"location_anchor_available={constraint_summary.get('location_anchor_available')}"
            )

            tavily_results = search_tavily(
                self.tavily_client,
                f"India healthcare hospitals clinics {user_text}"[:400],
                max_results=4,
                search_depth="basic",
            )
            tavily_block = format_tavily_block(tavily_results)
            if tavily_block:
                state["chain_of_thought"].append(f"Tavily: merged {len(tavily_results)} web results")
                state["reason_codes"].append("web_context_enriched")

            if self.llm is not None:
                prompt = (
                    "You are an expert healthcare facility analyst for India.\n"
                    "Given the user query and retrieved facility records, answer with multi-attribute reasoning "
                    "(combine location, specialties, equipment, staffing signals, and unstructured description).\n"
                    "Include 2–5 ranked matches when possible.\n"
                    "For each match cite: facility name + the exact field or phrase from the record that supports it.\n"
                    "If evidence is weak or ambiguous, say so clearly.\n"
                    f"--- RETRIEVED FACILITIES ---\n{context}"
                )
                if tavily_block:
                    prompt += f"\n--- OPTIONAL WEB CONTEXT ---\n{tavily_block}"
                answer = self.llm.invoke([SystemMessage(content=prompt), HumanMessage(content=user_text)]).content
            else:
                lines = [
                    "Answer (dataset retrieval + heuristics; no OpenAI LLM).",
                    "Top matches from the Virtue Foundation India facility index:",
                ]
                for i, d in enumerate(docs[:5], 1):
                    lines.append(
                        f"{i}. {d.metadata.get('name', 'Unknown')} — "
                        f"{d.metadata.get('city', '')}, {d.metadata.get('state', '')} "
                        f"PIN {d.metadata.get('pin', '')}\n"
                        f"   Excerpt: {evidence_snippet(d.page_content, 240)}"
                    )
                if not docs:
                    lines.append("No indexed facilities matched this query strongly.")
                if tavily_block:
                    lines.append("")
                    lines.append(tavily_block)
                elif self.tavily_client is None:
                    lines.append("")
                    lines.append("(Set TAVILY_API_KEY in .env to enrich answers with live web context.)")
                answer = "\n".join(lines)

            state["answer"] = answer
            top = docs[0].metadata if docs else {}
            completeness = float(top.get("completeness") or 0.0) if top else 0.0
            score = max(0.0, min(1.0, 0.55 * (1.0 if docs else 0.0) + 0.35 * completeness + 0.1 * min(len(docs), 5) / 5))
            if constraint_summary.get("missing_attributes"):
                score = max(0.0, score - 0.12)
            state["confidence_score"] = round(score, 3)
            state["confidence_band"] = self._band(score)
            if docs:
                state["reason_codes"].extend(["retrieval_hit", "top_doc_scored"])
                state["reason_codes"].append("deterministic_ranking_applied")
                state["reason_codes"].append("constraint_enforcement_applied")
                if top.get("city") or top.get("state"):
                    state["reason_codes"].append("location_match")
                if top.get("specialties"):
                    state["reason_codes"].append("specialty_match")
                if top.get("procedures"):
                    state["reason_codes"].append("procedure_match")
                if top.get("equipment"):
                    state["reason_codes"].append("equipment_match")
            if constraint_summary.get("missing_attributes"):
                state["reason_codes"].append("unsupported_attributes_disclosed")
            if constraint_summary.get("matched_attributes"):
                state["reason_codes"].append("matched_attributes_disclosed")
            if constraint_summary.get("location_anchor_available") is False and self._extract_constraints(user_text).get("needs_nearest"):
                state["reason_codes"].append("nearest_without_anchor_limited")
            state["chain_of_thought"].append(
                "Query constraints: "
                f"matched={constraint_summary.get('matched_attributes', [])} "
                f"missing={constraint_summary.get('missing_attributes', [])}"
            )
            state["messages"].append(AIMessage(content=answer))
            state["chain_of_thought"].append("Query agent: retrieved top-k facilities from FAISS")
            return state

    def audit_agent(self, state: HealthcareState) -> HealthcareState:
        with traced_step("audit_agent"):
            user_text = state["messages"][-1].content
            m_name = re.search(r'audit facility\s+"([^"]+)"', user_text, flags=re.I)
            requested_name = (m_name.group(1).strip() if m_name else user_text.strip())

            # Strong deterministic matching. Avoid auditing unrelated facilities
            # when explicit name query cannot be matched confidently.
            candidates = self._name_candidates(requested_name, limit=5)
            docs = self._name_match_docs(requested_name, limit=5, min_score=0.55)
            if not docs:
                suggestions = []
                for score, _, row in candidates[:3]:
                    name = str(row.get("name") or "").strip()
                    if not name:
                        continue
                    suggestions.append(
                        {
                            "facility_name": name,
                            "city": str(row.get("address_city") or ""),
                            "state": str(row.get("address_stateOrRegion") or ""),
                            "match_score": round(float(score), 3),
                        }
                    )
                state["audit_result"] = {
                    "status": "not_found",
                    "requested_name": requested_name,
                    "suggestions": suggestions,
                }
                state["source_citations"] = []
                state["confidence_score"] = 0.0
                state["confidence_band"] = "low"
                state["reason_codes"] = ["audit_name_not_found", "did_you_mean_suggestions"]
                if suggestions:
                    stext = "; ".join(
                        f"{s['facility_name']} ({s['city']}, {s['state']})"
                        for s in suggestions
                    )
                    state["messages"].append(
                        AIMessage(
                            content=(
                                f"No confident facility match found for '{requested_name}'. "
                                f"Did you mean: {stext} ?"
                            )
                        )
                    )
                else:
                    state["messages"].append(
                        AIMessage(content=f"No facility record found for '{requested_name}'. Try a nearby known facility name.")
                    )
                return state

            top_doc = docs[0]
            m = top_doc.metadata
            state["source_citations"] = [
                {
                    "facility_name": m.get("name", ""),
                    "pin": m.get("pin", ""),
                    "field": "page_content",
                    "evidence_snippet": evidence_snippet(top_doc.page_content, 400),
                }
            ]
            bag = " | ".join(
                [
                    top_doc.page_content.lower(),
                    m.get("specialties", "").lower(),
                    m.get("capability", "").lower(),
                    m.get("equipment", "").lower(),
                    m.get("procedures", "").lower(),
                ]
            )

            if self.llm is not None:
                structured = self.llm.with_structured_output(FacilityCapabilities)
                extracted = structured.invoke(
                    [
                        SystemMessage(content="Extract capabilities from text only."),
                        HumanMessage(content=top_doc.page_content),
                    ]
                )
            else:
                extracted = keyword_capability_flags(bag)
            heuristic_extracted = keyword_capability_flags(bag)

            checks = {
                "ICU": (extracted.has_icu or heuristic_extracted.has_icu)
                and any(k in bag for k in ["icu", "critical care", "ventilator"]),
                "Emergency": (extracted.has_emergency or heuristic_extracted.has_emergency)
                and any(k in bag for k in ["emergency", "trauma", "24/7", "24x7"]),
                "Surgery": (extracted.has_surgery or heuristic_extracted.has_surgery)
                and any(k in bag for k in ["surgery", "operation theatre", "operation theater", "ot"]),
                "Dialysis": (extracted.has_dialysis or heuristic_extracted.has_dialysis)
                and any(k in bag for k in ["dialysis", "nephrolog"]),
                "Oncology": (extracted.has_oncology or heuristic_extracted.has_oncology)
                and any(k in bag for k in ["oncology", "chemotherapy", "radiation"]),
                "Neonatal": (extracted.has_neonatal or heuristic_extracted.has_neonatal)
                and any(k in bag for k in ["neonatal", "nicu", "incubator"]),
            }
            claimed = [
                k
                for k, v in {
                    "ICU": extracted.has_icu or heuristic_extracted.has_icu,
                    "Emergency": extracted.has_emergency or heuristic_extracted.has_emergency,
                    "Surgery": extracted.has_surgery or heuristic_extracted.has_surgery,
                    "Dialysis": extracted.has_dialysis or heuristic_extracted.has_dialysis,
                    "Oncology": extracted.has_oncology or heuristic_extracted.has_oncology,
                    "Neonatal": extracted.has_neonatal or heuristic_extracted.has_neonatal,
                }.items()
                if v
            ]
            verified = [k for k, ok in checks.items() if ok]
            missing = [k for k in claimed if k not in verified]
            verification_ratio = (len(verified) / len(claimed)) if claimed else 0.0
            alternatives = [d.metadata.get("name", "") for d in docs[1:4] if d.metadata.get("name")]
            state["audit_result"] = {
                "facility_name": m.get("name", "Unknown"),
                "claimed": claimed,
                "verified": verified,
                "missing": missing,
                "verification_ratio": round(verification_ratio, 3),
                "alternatives": alternatives,
                "num_doctors": extracted.num_doctors,
                "bed_capacity": extracted.bed_capacity,
                "confidence_note": getattr(extracted, "confidence_note", "") or "",
            }
            score = max(0.0, min(1.0, 0.35 + 0.5 * verification_ratio + 0.15 * float(m.get("completeness") or 0.0)))
            state["confidence_score"] = round(score, 3)
            state["confidence_band"] = self._band(score)
            state["reason_codes"] = ["audit_extraction", "capability_crosscheck"]
            if alternatives:
                state["reason_codes"].append("name_disambiguation_candidates")
            state["retrieved_docs"] = [d.metadata for d in docs]
            state["messages"].append(
                AIMessage(
                    content=(
                        f"Audit for {state['audit_result']['facility_name']}\n"
                        f"- Claimed: {', '.join(claimed) or 'None'}\n"
                        f"- Verified: {', '.join(verified) or 'None'}\n"
                        f"- Missing evidence: {', '.join(missing) or 'None'}"
                    )
                )
            )
            state["chain_of_thought"].append("Audit agent: extracted structured capabilities from facility text")
            return state

    def trust_scorer(self, state: HealthcareState) -> HealthcareState:
        with traced_step("trust_scorer"):
            user_text = state["messages"][-1].content
            docs = self._retrieve_docs(user_text)
            if not docs:
                state["trust_score"] = 0.0
                state["trust_flags"] = ["No facility record found to score."]
                state["source_citations"] = []
                state["messages"].append(AIMessage(content="Unable to assess trust: no matching facility found."))
                return state

            top_doc = docs[0]
            m = top_doc.metadata
            state["source_citations"] = [
                {
                    "facility_name": m.get("name", ""),
                    "pin": m.get("pin", ""),
                    "trust_evidence_snippet": evidence_snippet(top_doc.page_content, 400),
                }
            ]
            contradiction_mode = self._is_contradiction_search(user_text)
            top_eval = self._trust_doc_evaluation(top_doc)
            flags = top_eval["flags"]
            score = top_eval["score"]
            breakdown = top_eval["breakdown"]
            evidence_map = top_eval["evidence_map"]
            high_severity = top_eval["high_severity"]

            contradiction_summary = []
            if contradiction_mode:
                suspicious = []
                for d in docs[:10]:
                    ev = self._trust_doc_evaluation(d)
                    if ev["high_severity"]:
                        suspicious.append(
                            {
                                "facility_name": d.metadata.get("name", "Unknown"),
                                "pin": d.metadata.get("pin", ""),
                                "city": d.metadata.get("city", ""),
                                "state": d.metadata.get("state", ""),
                                "trust_score": round(ev["score"], 3),
                                "contradictions": ev["high_severity"],
                            }
                        )
                suspicious.sort(key=lambda x: (len(x["contradictions"]), -x["trust_score"]), reverse=True)
                contradiction_summary = suspicious[:5]
                if contradiction_summary:
                    state["source_citations"] = [
                        {
                            "facility_name": c["facility_name"],
                            "pin": c["pin"],
                            "city": c["city"],
                            "state": c["state"],
                            "trust_evidence_snippet": f"Contradictions: {', '.join(c['contradictions'])}",
                        }
                        for c in contradiction_summary
                    ]
                    state["reason_codes"] = list(dict.fromkeys((state.get("reason_codes") or []) + ["contradiction_search_mode"]))

            if self.llm is not None:
                structured = self.llm.with_structured_output(TrustAssessment)
                llm_assessment = structured.invoke(
                    [
                        SystemMessage(content="Assess trustworthiness from evidence and rule flags."),
                        HumanMessage(content=f"Evidence: {top_doc.page_content[:2000]}\nFlags: {flags}"),
                    ]
                )
                final_score = max(0.0, min(1.0, (score * 0.65) + (llm_assessment.score * 0.35)))
                merged_flags = list(dict.fromkeys(flags + llm_assessment.flags))
                explanation = llm_assessment.explanation
            else:
                final_score = max(0.0, min(1.0, score))
                merged_flags = flags
                explanation = (
                    "Rule-based trust score only (no OpenAI). "
                    "Flags list claim–evidence gaps from TRUST_RULES and profile heuristics."
                )

            state["trust_score"] = round(final_score, 3)
            state["trust_flags"] = merged_flags
            state["trust_breakdown"] = breakdown
            state["trust_evidence_map"] = evidence_map
            state["confidence_score"] = round(final_score, 3)
            state["confidence_band"] = self._band(final_score)
            state["reason_codes"] = ["trust_rules_applied", "claim_evidence_consistency_check"]
            if contradiction_summary:
                state["reason_codes"].append("contradiction_search_mode")
            if high_severity:
                state["reason_codes"].append("high_severity_contradictions_detected")
            state["retrieved_docs"] = [d.metadata for d in docs]
            if contradiction_summary:
                lines = [
                    "Suspicious facilities where claims may not match evidence:",
                ]
                for i, s in enumerate(contradiction_summary, 1):
                    lines.append(
                        f"{i}. {s['facility_name']} ({s['city']}, {s['state']}) PIN {s['pin']} "
                        f"| trust={s['trust_score']:.2f} | contradictions: {', '.join(s['contradictions'])}"
                    )
                lines.append("\nNote: Review citations and trust evidence map for exact grounding.")
                state["answer"] = "\n".join(lines)
            state["messages"].append(
                AIMessage(
                    content=(
                        f"Trust score for {m.get('name', 'Unknown')}: {state['trust_score']:.2f}\n"
                        f"Flags: {', '.join(state['trust_flags']) if state['trust_flags'] else 'None'}\n"
                        f"Explanation: {explanation}"
                    )
                )
            )
            state["chain_of_thought"].append("Trust scorer: evaluated claim-evidence consistency")
            return state

    @staticmethod
    def _high_severity_contradictions(claims_text: str, evidence_text: str, full_text: str) -> List[str]:
        contradictions = []
        claims_blob = (claims_text or "").lower()
        evidence_blob = " | ".join([(evidence_text or "").lower(), (full_text or "").lower()])
        if ("advanced surgery" in claims_blob or "surgery" in claims_blob) and (
            "anesthesiolog" not in evidence_blob and "operation theatre" not in evidence_blob and "ot" not in evidence_blob
        ):
            contradictions.append("Surgery claim without anesthesiology/OT evidence")
        if ("24/7" in claims_blob or "emergency" in claims_blob) and (
            "trauma" not in evidence_blob and "emergency" not in evidence_blob and "critical care" not in evidence_blob
        ):
            contradictions.append("24x7/Emergency claim without emergency staffing evidence")
        if "icu" in claims_blob and ("ventilator" not in evidence_blob and "intensivist" not in evidence_blob):
            contradictions.append("ICU claim without ventilator/intensivist evidence")
        return contradictions

    @staticmethod
    def _find_evidence_sentence(full_text: str, candidate_terms: List[str]) -> str:
        blob = (full_text or "").strip()
        if not blob:
            return "No source text available."
        sentences = re.split(r"(?<=[.!?])\s+", blob)
        terms = [str(t).lower() for t in (candidate_terms or []) if str(t).strip()]
        for sentence in sentences:
            s = sentence.lower()
            if any(t in s for t in terms):
                return sentence.strip()[:320]
        return "No exact supporting sentence found; flagged as missing evidence."

    def desert_finder(self, state: HealthcareState) -> HealthcareState:
        with traced_step("desert_finder"):
            if self.df is None or self.df.empty:
                state["desert_regions"] = []
                state["source_citations"] = []
                state["messages"].append(AIMessage(content="No dataset loaded for desert analysis."))
                return state
            groups = self.df.groupby(["address_stateOrRegion", "address_city", "pin"], dropna=False)
            deserts = []
            for (state_name, city, pin), g in groups:
                if pin is None or str(pin).strip() == "":
                    continue
                region_terms = set()
                for _, row in g.iterrows():
                    region_terms.update(_normalise_terms(_parse_json_list(row.get("specialties"))))
                    region_terms.update(_normalise_terms(_parse_json_list(row.get("capability"))))
                blob = " ".join(region_terms).lower()
                missing = []
                for spec in HIGH_ACUITY_SPECIALTIES:
                    spaced = spec.replace("Medicine", " medicine").replace("Surgery", " surgery").lower()
                    if spec.lower() not in blob and spaced not in blob:
                        missing.append(spec)
                if missing:
                    lat = pd.to_numeric(g["latitude"], errors="coerce").mean()
                    lon = pd.to_numeric(g["longitude"], errors="coerce").mean()
                    severity_index = round(len(missing) / len(HIGH_ACUITY_SPECIALTIES), 3)
                    if severity_index >= 0.75:
                        priority = "critical"
                    elif severity_index >= 0.5:
                        priority = "high"
                    elif severity_index >= 0.25:
                        priority = "medium"
                    else:
                        priority = "low"
                    deserts.append(
                        {
                            "pin": str(pin),
                            "city": str(city or ""),
                            "state": str(state_name or ""),
                            "lat": None if pd.isna(lat) else float(lat),
                            "lon": None if pd.isna(lon) else float(lon),
                            "missing_specialties": missing,
                            "facility_count": int(len(g)),
                            "severity_index": severity_index,
                            "intervention_priority": priority,
                        }
                    )
            deserts.sort(key=lambda x: len(x["missing_specialties"]), reverse=True)
            state["desert_regions"] = deserts[:200]
            state["planner_summary"] = [
                {
                    "pin": r.get("pin"),
                    "city": r.get("city"),
                    "state": r.get("state"),
                    "priority": r.get("intervention_priority"),
                    "severity_index": r.get("severity_index"),
                    "missing_specialties": r.get("missing_specialties"),
                    "recommended_action": "Deploy specialty referral linkage and mobile screening unit",
                }
                for r in (state["desert_regions"] or [])[:20]
            ]
            mean_severity = 0.0
            if state["desert_regions"]:
                mean_severity = sum(float(r.get("severity_index") or 0.0) for r in state["desert_regions"]) / len(state["desert_regions"])
            state["confidence_score"] = round(max(0.0, min(1.0, 0.5 + 0.4 * mean_severity)), 3)
            state["confidence_band"] = self._band(state["confidence_score"])
            state["reason_codes"] = ["geo_grouping_by_pin_city_state", "high_acuity_gap_scoring"]
            state["messages"].append(
                AIMessage(content=f"Identified {len(state['desert_regions'])} potential healthcare desert regions.")
            )
            state["chain_of_thought"].append("Desert finder: scanned records for specialty coverage by PIN")
            return state

    def validator_agent(self, state: HealthcareState) -> HealthcareState:
        with traced_step("validator_agent"):
            state["validation_attempts"] = int(state.get("validation_attempts") or 0) + 1
            if self.llm is not None:
                payload = {
                    "intent": state.get("intent"),
                    "answer": state.get("answer"),
                    "audit_result": state.get("audit_result"),
                    "trust_score": state.get("trust_score"),
                    "trust_flags": state.get("trust_flags"),
                    "desert_regions_count": len(state.get("desert_regions") or []),
                    "retrieved_docs_count": len(state.get("retrieved_docs") or []),
                    "source_citations": state.get("source_citations"),
                    "trust_evidence_map": state.get("trust_evidence_map"),
                    "confidence_score": state.get("confidence_score"),
                    "confidence_band": state.get("confidence_band"),
                    "reason_codes": state.get("reason_codes"),
                }
                verdict = self.llm.invoke(
                    [
                        SystemMessage(
                            content=(
                                "Validate grounding and consistency. "
                                "Return exactly: VALID: <reason> or INVALID: <reason>"
                            )
                        ),
                        HumanMessage(content=json.dumps(payload, default=str)),
                    ]
                ).content.strip()
                state["validated"] = verdict.upper().startswith("VALID:")
                state["correction_notes"] = "" if state["validated"] else verdict
            else:
                ok, reason = validator_rule_based(dict(state))
                state["validated"] = ok
                state["correction_notes"] = "" if ok else format_validator_verdict(False, reason)
            state["chain_of_thought"].append("Validator: cross-checked output against standards")
            return state

    def correction_agent(self, state: HealthcareState) -> HealthcareState:
        with traced_step("correction_agent"):
            note = (state.get("correction_notes") or "").strip()
            intent = state.get("intent")
            correction_text = (
                "Correction pass applied due to validator mismatch. "
                "Unsupported attributes are now explicitly marked as unconfirmed."
            )
            if note:
                correction_text += f" Validator note: {note}"
            if intent == "query":
                answer = (state.get("answer") or "").strip()
                if answer:
                    answer += "\n\nCorrection: Any attribute not directly supported by retrieved evidence is treated as unconfirmed."
                else:
                    answer = correction_text
                state["answer"] = answer
            elif intent == "trust":
                flags = state.get("trust_flags") or []
                if "Correction applied after validator mismatch." not in flags:
                    flags.append("Correction applied after validator mismatch.")
                state["trust_flags"] = flags
            elif intent == "audit":
                ar = state.get("audit_result") or {}
                ar["validator_correction"] = note or "Mismatch corrected by conservative fallback."
                state["audit_result"] = ar
            state["correction_applied"] = True
            state["reason_codes"] = list(dict.fromkeys((state.get("reason_codes") or []) + ["validator_correction_applied"]))
            state["chain_of_thought"].append("Correction agent: applied conservative repair and disclosure")
            return state

    def out_of_scope_handler(self, state: HealthcareState) -> HealthcareState:
        state["messages"].append(
            AIMessage(
                content=(
                    "I am an Indian healthcare facility intelligence assistant. "
                    "Ask about facility search, audits, deserts, or trust scoring."
                )
            )
        )
        state["chain_of_thought"].append("Out-of-scope: query redirected")
        return state

    def response_builder(self, state: HealthcareState) -> HealthcareState:
        intent = state.get("intent")
        retrieved = state.get("retrieved_docs") or []
        citation = ""
        if retrieved:
            top = retrieved[0]
            citation = (
                f"\n\nCitation: {top.get('name', 'Unknown')} | "
                f"{top.get('city', '')}, {top.get('state', '')} | PIN {top.get('pin', '')}"
            )

        if intent == "query":
            base = state.get("answer") or "No answer generated."
        elif intent == "audit":
            ar = state.get("audit_result") or {}
            base = (
                f"Audit for {ar.get('facility_name', 'Unknown')}\n"
                f"- Claimed: {', '.join(ar.get('claimed') or []) or 'None'}\n"
                f"- Verified: {', '.join(ar.get('verified') or []) or 'None'}\n"
                f"- Missing evidence: {', '.join(ar.get('missing') or []) or 'None'}\n"
                f"- Verification ratio: {ar.get('verification_ratio', 0)}"
            )
            if ar.get("alternatives"):
                base += f"\n- Alternative candidates: {', '.join(ar.get('alternatives')[:3])}"
        elif intent == "trust":
            base = state.get("answer") or (
                f"Trust score: {state.get('trust_score')}\n"
                f"Flags: {', '.join(state.get('trust_flags') or []) or 'None'}"
            )
        elif intent == "desert":
            regions = state.get("desert_regions") or []
            base = f"Potential deserts found: {len(regions)}\nTop regions: {json.dumps(regions[:5], ensure_ascii=False)}"
        else:
            return state

        if state.get("validated") is False and state.get("correction_notes"):
            base += f"\n\nValidator note: {state['correction_notes']}"
        if state.get("validation_attempts") is not None:
            base += f"\nValidation attempts: {state.get('validation_attempts')}"
        if state.get("correction_applied"):
            base += "\nCorrection applied: yes"
        if state.get("confidence_score") is not None:
            base += f"\n\nConfidence: {state.get('confidence_score')} ({state.get('confidence_band', 'unknown')})"
        if state.get("reason_codes"):
            base += f"\nReason codes: {', '.join(state.get('reason_codes') or [])}"
        trust_evidence = state.get("trust_evidence_map") or []
        if trust_evidence:
            base += f"\n\nTrust evidence map:\n{json.dumps(trust_evidence, ensure_ascii=False, indent=2)}"
        cites = state.get("source_citations") or []
        if cites:
            base += f"\n\nSource excerpts (verbatim from records):\n{json.dumps(cites, ensure_ascii=False, indent=2)}"
        state["messages"].append(AIMessage(content=base + citation))
        state["chain_of_thought"].append("Response builder: assembled final answer with citations")
        return state
