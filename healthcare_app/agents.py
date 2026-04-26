import json
import math
import re
from typing import Any, List, Optional

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from healthcare_app.citations import evidence_snippet
from healthcare_app.config import HIGH_ACUITY_SPECIALTIES, TRUST_RULES
from healthcare_app.data import _parse_json_list, format_docs_for_llm
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
            docs = [r[0] for r in ranked]
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
                for i, (_, score, reasons) in enumerate(ranked[:5], 1):
                    ranking_lines.append(f"[{i}] score={score:.3f} | reasons={', '.join(reasons)}")
                context = f"{context}\n\n--- DETERMINISTIC RANKING SIGNALS ---\n" + "\n".join(ranking_lines)

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
            state["confidence_score"] = round(score, 3)
            state["confidence_band"] = self._band(score)
            if docs:
                state["reason_codes"].extend(["retrieval_hit", "top_doc_scored"])
                state["reason_codes"].append("deterministic_ranking_applied")
                if top.get("city") or top.get("state"):
                    state["reason_codes"].append("location_match")
                if top.get("specialties"):
                    state["reason_codes"].append("specialty_match")
                if top.get("procedures"):
                    state["reason_codes"].append("procedure_match")
                if top.get("equipment"):
                    state["reason_codes"].append("equipment_match")
            state["messages"].append(AIMessage(content=answer))
            state["chain_of_thought"].append("Query agent: retrieved top-k facilities from FAISS")
            return state

    def audit_agent(self, state: HealthcareState) -> HealthcareState:
        with traced_step("audit_agent"):
            user_text = state["messages"][-1].content
            docs = self._retrieve_docs(user_text)
            if not docs:
                state["audit_result"] = {"status": "not_found"}
                state["source_citations"] = []
                state["confidence_score"] = 0.0
                state["confidence_band"] = "low"
                state["reason_codes"] = ["no_retrieval_hit"]
                state["messages"].append(AIMessage(content="No facility record found to audit."))
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

            checks = {
                "ICU": extracted.has_icu and any(k in bag for k in ["icu", "critical care", "ventilator"]),
                "Emergency": extracted.has_emergency and any(k in bag for k in ["emergency", "trauma", "24/7"]),
                "Surgery": extracted.has_surgery and any(k in bag for k in ["surgery", "operation theatre", "ot"]),
                "Dialysis": extracted.has_dialysis and any(k in bag for k in ["dialysis", "nephrolog"]),
                "Oncology": extracted.has_oncology and any(k in bag for k in ["oncology", "chemotherapy", "radiation"]),
                "Neonatal": extracted.has_neonatal and any(k in bag for k in ["neonatal", "nicu", "incubator"]),
            }
            claimed = [
                k
                for k, v in {
                    "ICU": extracted.has_icu,
                    "Emergency": extracted.has_emergency,
                    "Surgery": extracted.has_surgery,
                    "Dialysis": extracted.has_dialysis,
                    "Oncology": extracted.has_oncology,
                    "Neonatal": extracted.has_neonatal,
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
            claims = _parse_json_list(m.get("capability", "")) + _parse_json_list(m.get("specialties", ""))
            claims_text = " | ".join(claims).lower()
            evidence_text = " | ".join(
                [top_doc.page_content.lower(), m.get("equipment", "").lower(), m.get("procedures", "").lower()]
            )
            flags = []
            score = 1.0
            breakdown = []
            for claim_name, required_terms in TRUST_RULES.items():
                ck = claim_name.lower()
                if ck.replace(" ", "") in claims_text.replace(" ", "") or ck in claims_text:
                    missing_terms = [t for t in required_terms if t.lower() not in evidence_text]
                    if missing_terms:
                        flags.append(f"{claim_name}: missing evidence -> {', '.join(missing_terms)}")
                        score -= 0.12
                        breakdown.append({"type": "claim_evidence_gap", "claim": claim_name, "penalty": 0.12})
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
            state["confidence_score"] = round(final_score, 3)
            state["confidence_band"] = self._band(final_score)
            state["reason_codes"] = ["trust_rules_applied", "claim_evidence_consistency_check"]
            state["retrieved_docs"] = [d.metadata for d in docs]
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
            base = f"Audit result: {json.dumps(state.get('audit_result') or {}, ensure_ascii=False)}"
        elif intent == "trust":
            base = (
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
        if state.get("confidence_score") is not None:
            base += f"\n\nConfidence: {state.get('confidence_score')} ({state.get('confidence_band', 'unknown')})"
        if state.get("reason_codes"):
            base += f"\nReason codes: {', '.join(state.get('reason_codes') or [])}"
        cites = state.get("source_citations") or []
        if cites:
            base += f"\n\nSource excerpts (verbatim from records):\n{json.dumps(cites, ensure_ascii=False, indent=2)}"
        state["messages"].append(AIMessage(content=base + citation))
        state["chain_of_thought"].append("Response builder: assembled final answer with citations")
        return state
