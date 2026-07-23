from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import unicodedata
import segmentation
import token_utils


@dataclass
class DiscourseState:
    topic_value: Optional[str] = None
    topic_conf: float = 0.0
    last_agent_value: Optional[str] = None
    last_agent_conf: float = 0.0
    last_patient_value: Optional[str] = None
    last_patient_conf: float = 0.0
    pt_map: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def decay_clause(self) -> None:
        self.topic_conf *= 0.85
        self.last_agent_conf *= 0.85
        self.last_patient_conf *= 0.85
        for _, entry in self.pt_map.items():
            if "conf" in entry:
                entry["conf"] *= 0.85

    def reset_after_boundary(self) -> None:
        self.pt_map.clear()
        self.last_agent_conf = 0.0
        self.last_patient_conf = 0.0
        self.topic_conf *= 0.90


def parse_clause(
    tokens: List[Dict[str, Any]],
    state: DiscourseState,
    boundary_seen: bool,
    split_meta: Optional[Dict[str, Any]] = None,
    clause_flags: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    action = {"value": None, "conf": 0.0, "rule": None, "flags": {}}
    agent = {"value": None, "conf": 0.0, "rule": None, "flags": {}}
    patient = {"value": None, "conf": 0.0, "rule": None, "flags": {}}

    debug = {
        "verb_index": None,
        "verb_count": 0,
        "noun_count": 0,
        "has_pt": False,
        "pt_map_size": 0,
        "pt_resolved": False,
        "postverb_pt_key": None,
        "postverb_pt_source": None,
        "verb_transitive": None,
        "verb_valence": None,
        "verb_transitive_state": None,
        "preverb_is_noun": False,
        "postverb_is_pt": False,
        "rule_trace": [],
        "conflict": False,
        "warning": False,
        "needs_ml": False,
        "self_ref_repaired": None,
        "self_ref_conflict": False,
        "best_candidate_conf": 0.0,
        "candidates_agent": [],
        "candidates_patient": [],
        "agent_candidates": [],
        "patient_candidates": [],
        "baseline_agent": None,
        "baseline_patient": None,
        "agent_rule": None,
        "needs_ml_agent": False,
        "needs_ml_patient": False,
        "ml_reason_agent": None,
        "ml_reason_patient": None,
        "r30_trigger": None,
        "boundary_reset": None,
        "boundary_decay": None,
    }

    def _detect_boundary_strength(
        tokens_list: List[Dict[str, Any]],
        split_meta_ref: Optional[Dict[str, Any]],
        clause_flags_ref: Optional[Dict[str, Any]],
        boundary_seen_ref: bool,
    ) -> Optional[str]:
        if clause_flags_ref:
            strength = clause_flags_ref.get("boundary_strength")
            if strength in {"strong", "weak"}:
                return strength
        if split_meta_ref and split_meta_ref.get("reason") == "R26_BOUNDARY":
            strength = (split_meta_ref.get("flags") or {}).get("strength")
            if strength in {"strong", "weak"}:
                return strength
        for tok in tokens_list:
            modifier = tok.get("modifier") or {}
            boundary_flag = str(modifier.get("boundary") or "").strip().lower()
            if boundary_flag in {"strong", "1", "true"}:
                return "strong"
            if boundary_flag == "weak":
                return "weak"
            word_norm = token_utils.norm_nfkc(tok.get("word", ""))
            if word_norm in {"stop", "finish", "end", "?"}:
                return "strong"
            if word_norm in {"、", "…", "...", ",", "。", ";", ":"}:
                return "weak"
        if boundary_seen_ref:
            return "strong"
        return None

    boundary_strength = _detect_boundary_strength(
        tokens, split_meta, clause_flags, boundary_seen
    )
    if boundary_strength == "strong":
        state.last_agent_conf = 0.0
        state.last_patient_conf = 0.0
        state.last_agent_value = None
        state.last_patient_value = None
        state.pt_map.clear()
        debug["boundary_reset"] = "strong"
    elif boundary_strength == "weak":
        decay = 0.5
        state.last_agent_conf *= decay
        state.last_patient_conf *= decay
        debug["boundary_reset"] = "weak"
        debug["boundary_decay"] = decay
    else:
        state.decay_clause()

    def resolve_pt(tok: Dict[str, Any], state_ref: DiscourseState):
        ref = token_utils.pt_inline_referent(tok.get("word", ""))
        if ref:
            return ref, 0.90, "PT_INLINE"

        mods_raw = tok.get("mods_raw") or []
        if mods_raw:
            value = mods_raw[0]
            if isinstance(value, str) and value.strip():
                return value, 0.95, "DIRECT_MODS_RAW"

        modifier = tok.get("modifier") or {}
        for key in ["referent", "ref", "pt_ref", "entity", "resolved_value", "pt_value"]:
            value = modifier.get(key)
            if isinstance(value, str) and value.strip():
                return value, 0.95, "DIRECT_MODIFIER"

        inline_ref = token_utils.pt_inline_referent(tok.get("word"))
        if inline_ref:
            return inline_ref, 0.95, "INLINE_PT"

        pt_key = token_utils.pt_key(tok)
        if pt_key == "pt1":
            return "SELF", 0.90, "DEFAULT_PT1"
        if pt_key == "pt2":
            return "ADDRESSEE", 0.90, "DEFAULT_PT2"
        if pt_key and pt_key in state_ref.pt_map:
            return state_ref.pt_map[pt_key].get("value"), 0.85, "PT_MAP"

        return None, 0.0, "UNRESOLVED"

    # R1: Verb as Core
    verb_index = -1
    verb_count = 0
    noun_count = 0
    has_pt = False
    noun_tokens = []
    content_tokens = []
    for idx, tok in enumerate(tokens):
        if token_utils.is_pt(tok):
            has_pt = True
        elif token_utils.is_noun(tok):
            noun_count += 1
            noun_tokens.append(tok)
            content_tokens.append(tok)
        elif token_utils.is_adj(tok):
            content_tokens.append(tok)

        if segmentation.is_verb_lexical(tok):
            verb_count += 1
            verb_index = idx

    debug["verb_index"] = verb_index
    debug["verb_count"] = verb_count
    debug["noun_count"] = noun_count
    debug["has_pt"] = has_pt

    verb_tok = tokens[verb_index] if verb_index >= 0 else None
    verb_valence = None
    if verb_tok:
        verb_valence = (verb_tok.get("modifier") or {}).get("valence")
        if verb_valence in {"transitive", "ambitransitive"}:
            debug["verb_transitive"] = True
        elif verb_valence == "intransitive":
            debug["verb_transitive"] = False
        else:
            debug["verb_transitive"] = None
        debug["verb_valence"] = verb_valence
        debug["verb_transitive_state"] = debug["verb_transitive"]

    if verb_index == -1:
        action["value"] = "UNKNOWN"
        action["conf"] = 0.0
        action["rule"] = "NO_VERB"
    else:
        tok = tokens[verb_index]
        action["value"] = tok.get("word")
        action["conf"] = 0.95
        action["rule"] = "R1"
        action["flags"] = {
            "is_verb": True,
            "verb_lexical": True,
            "verb_index": verb_index,
            "verb_valence": tok.get("modifier", {}).get("valence"),
        }
        debug["rule_trace"].append("R1")
        debug["best_candidate_conf"] = action["conf"]

    # R5: topic from first nominal token
    for idx, tok in enumerate(tokens):
        if verb_index >= 0 and idx >= verb_index:
            break
        if token_utils.is_noun(tok) and not token_utils.is_pt(tok):
            state.topic_value = tok.get("word")
            state.topic_conf = max(state.topic_conf, 0.70)
            debug["rule_trace"].append("R5_TOPIC")
            break

    unknown_prior = 0.10

    def _build_candidates(state_ref: DiscourseState) -> List[Dict[str, Any]]:
        return [
            {
                "source": "TOPIC",
                "value": state_ref.topic_value,
                "conf": state_ref.topic_conf if state_ref.topic_value is not None else 0.0,
            },
            {
                "source": "LAST_AGENT",
                "value": state_ref.last_agent_value,
                "conf": state_ref.last_agent_conf
                if state_ref.last_agent_value is not None
                else 0.0,
            },
            {
                "source": "LAST_PATIENT",
                "value": state_ref.last_patient_value,
                "conf": state_ref.last_patient_conf
                if state_ref.last_patient_value is not None
                else 0.0,
            },
            {
                "source": "UNKNOWN",
                "value": "UNKNOWN",
                "conf": unknown_prior,
            },
        ]

    def _baseline_from_candidates(
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        available = [candidate for candidate in candidates if candidate["value"] is not None]
        if not available:
            return {"source": "UNKNOWN", "value": "UNKNOWN", "conf": unknown_prior}
        return max(available, key=lambda candidate: candidate["conf"])

    candidates = _build_candidates(state)
    debug["agent_candidates"] = [dict(candidate) for candidate in candidates]
    debug["patient_candidates"] = [dict(candidate) for candidate in candidates]
    debug["candidates_agent"] = [
        (candidate["value"], candidate["conf"], candidate["source"])
        for candidate in candidates
    ]
    debug["candidates_patient"] = [
        (candidate["value"], candidate["conf"], candidate["source"])
        for candidate in candidates
    ]

    # R22/R6: update PT map and resolve PT tokens, left-to-right
    resolved_pts = []
    for idx, tok in enumerate(tokens):
        prev_tok = tokens[idx - 1] if idx > 0 else None
        if prev_tok and token_utils.is_noun(prev_tok) and token_utils.is_pt(tok):
            pt_key = token_utils.pt_key(tok)
            if pt_key and pt_key != "pt":
                state.pt_map[pt_key] = {"value": prev_tok.get("word"), "conf": 0.95}
                debug["rule_trace"].append("R22_ASSIGN")

        if token_utils.is_pt(tok):
            resolved_value, resolved_conf, source = resolve_pt(tok, state)
            if source == "DIRECT_MODS_RAW":
                debug["rule_trace"].append("R6_RESOLVE_MODS_RAW")
            elif source == "DIRECT_MODIFIER":
                debug["rule_trace"].append("R6_RESOLVE_MODIFIER")
            elif source == "PT_MAP":
                debug["rule_trace"].append("R6_RESOLVE_MAP")
            else:
                debug["rule_trace"].append("R13_PT_AMBIGUOUS_CANDIDATE")
            resolved_pts.append(
                {
                    "index": idx,
                    "raw_word": tok.get("word"),
                    "pt_key": token_utils.pt_key(tok),
                    "resolved_value": resolved_value,
                    "resolved_conf": resolved_conf,
                    "source": source,
                }
            )

    postverb_resolved_value = None
    postverb_resolved_conf = 0.0
    if verb_index >= 0 and verb_index + 1 < len(tokens):
        tok_post = tokens[verb_index + 1]
        if token_utils.is_pt(tok_post):
            pt_ref = (tok_post.get("modifier") or {}).get("pt_ref")
            if pt_ref:
                debug["postverb_is_pt"] = True
                debug["pt_resolved"] = True
                debug["postverb_pt_source"] = "PT_REF"
                patient["value"] = pt_ref
                patient["conf"] = 0.95
                patient["rule"] = "R12"
                patient["flags"] = {
                    "postverb_is_pt": True,
                    "pt_ref": pt_ref,
                    "pt_resolved": True,
                }
                debug["needs_ml_patient"] = False
                debug["block_patient_fill"] = True
            else:
                debug["postverb_is_pt"] = True
                debug["postverb_pt_key"] = token_utils.pt_key(tok_post)
                (
                    postverb_resolved_value,
                    postverb_resolved_conf,
                    postverb_source,
                ) = resolve_pt(tok_post, state)
                debug["postverb_pt_source"] = postverb_source
                debug["pt_resolved"] = postverb_resolved_conf > 0.0

    debug["pt_map_size"] = len(state.pt_map)
    debug["pt_resolutions"] = resolved_pts

    # R12/R13/R11/R10: assign patient
    if debug["postverb_is_pt"] and patient["rule"] is None:
        if postverb_resolved_conf > 0.0:
            patient["value"] = postverb_resolved_value
            patient["conf"] = 0.95
            patient["rule"] = "R12"
            patient["flags"] = {
                "postverb_is_pt": True,
                "pt_key": debug["postverb_pt_key"],
                "pt_resolved": True,
            }
            debug["needs_ml_patient"] = False
            debug["pt_resolved"] = True
            debug["postverb_pt_source"] = postverb_source
            debug["rule_trace"].append("R12")
        else:
            patient["value"] = "UNKNOWN"
            patient["conf"] = 0.0
            patient["rule"] = "R13"
            tok_post = tokens[verb_index + 1] if verb_index + 1 < len(tokens) else {}
            pt_word = (tok_post.get("word") or "")
            pt_norm = unicodedata.normalize("NFKC", str(pt_word)).strip().lower()
            pt_ref = (tok_post.get("modifier") or {}).get("pt_ref")
            is_pt_bare = pt_norm == "pt" and not pt_ref
            if is_pt_bare:
                patient["value"] = None
                patient["conf"] = 0.0
                patient["rule"] = None
                debug["pt_is_bare"] = True
                debug["r13_reason"] = "PT_BARE_NONBLOCKING"
                debug["rule_trace"].append("R13_PT_BARE_NONBLOCKING")
            else:
                debug["r13_reason"] = "PT_UNRESOLVED_BLOCKING"
                debug["block_patient_fill"] = True

    if patient["rule"] is None:
        preverb_is_noun = False
        if (
            verb_index >= 1
            and token_utils.is_noun(tokens[verb_index - 1])
            and not segmentation.is_verb_lexical(tokens[verb_index - 1])
        ):
            preverb_is_noun = True
            debug["preverb_is_noun"] = True
        if debug["verb_transitive"] is not False and preverb_is_noun:
            patient["value"] = tokens[verb_index - 1].get("word")
            patient["conf"] = 0.95 if debug["verb_transitive"] else 0.80
            patient["rule"] = "R11"
            debug["needs_ml_patient"] = False

    if patient["rule"] is None:
        if debug["verb_transitive"] is not False and noun_count == 1 and not has_pt:
            patient["value"] = noun_tokens[0].get("word") if noun_tokens else None
            patient["conf"] = 0.90 if debug["verb_transitive"] else 0.75
            patient["rule"] = "R10"
            debug["needs_ml_patient"] = False

    # R3: implicit copula when no verb and two content tokens
    if verb_count == 0 and len(content_tokens) == 2:
        action["value"] = "BE"
        action["conf"] = 0.70
        action["rule"] = "R3"
        if agent["rule"] is None:
            agent["value"] = content_tokens[0].get("word")
            agent["conf"] = 0.70
            agent["rule"] = "R3"
        if patient["rule"] is None:
            patient["value"] = content_tokens[1].get("word")
            patient["conf"] = 0.70
            patient["rule"] = "R3"

    if agent["rule"] is None:
        baseline_agent = _baseline_from_candidates(candidates)
        agent["value"] = baseline_agent["value"]
        agent["conf"] = baseline_agent["conf"]
        agent["rule"] = "R8"
        debug["agent_rule"] = "R8"
        debug["baseline_agent"] = dict(baseline_agent)
        if baseline_agent["conf"] < 0.70:
            debug["needs_ml_agent"] = True
            debug["ml_reason_agent"] = "low_conf"

    if (
        agent["value"] == "UNKNOWN"
        and agent["conf"] == 0.0
        and verb_count > 0
        and noun_count == 0
    ):
        agent["value"] = "SELF"
        agent["conf"] = 0.60
        agent["rule"] = "R9"
        agent["flags"]["needs_ml"] = True
        debug["rule_trace"].append("R9_SELF")
        debug["needs_ml_agent"] = True
        debug["ml_reason_agent"] = "low_conf"

    if patient["rule"] is None and not debug.get("block_patient_fill", False):
        baseline_patient = _baseline_from_candidates(candidates)
        patient["value"] = baseline_patient["value"]
        patient["conf"] = baseline_patient["conf"]
        patient["rule"] = "R8"
        debug["baseline_patient"] = dict(baseline_patient)
        if baseline_patient["conf"] < 0.70:
            debug["needs_ml_patient"] = True
            debug["ml_reason_patient"] = "low_conf"

    # R23: no self-reference without reflexive marker
    def _norm_word(value: Any) -> str:
        return str(value).strip().lower() if value is not None else ""

    reflexive_markers = {"self", "\u81ea\u5206"}
    has_reflexive_marker = any(
        _norm_word(tok.get("word")) in reflexive_markers for tok in tokens
    )

    if (
        agent["value"] not in {None, "UNKNOWN"}
        and patient["value"] not in {None, "UNKNOWN"}
        and agent["value"] == patient["value"]
        and not has_reflexive_marker
    ):
        debug["self_ref_conflict"] = True
        debug["needs_ml"] = True
        if agent["rule"] == "R8" and patient["rule"] != "R8":
            agent["value"] = "UNKNOWN"
            agent["conf"] = 0.0
            agent["rule"] = "R23_DROP_AGENT_R8"
            debug["self_ref_repaired"] = "drop_agent_r8"
        elif patient["rule"] == "R8" and agent["rule"] != "R8":
            patient["value"] = "UNKNOWN"
            patient["conf"] = 0.0
            patient["rule"] = "R23_DROP_PATIENT_R8"
            debug["self_ref_repaired"] = "drop_patient_r8"
        else:
            drop_patient = patient["conf"] <= agent["conf"]
            if drop_patient:
                patient["value"] = "UNKNOWN"
                patient["conf"] = 0.0
                patient["rule"] = "R23_DROP_WEAKER"
            else:
                agent["value"] = "UNKNOWN"
                agent["conf"] = 0.0
                agent["rule"] = "R23_DROP_WEAKER"
            debug["self_ref_repaired"] = "drop_weaker"

    debug["best_candidate_conf"] = max(agent["conf"], patient["conf"])

    # R30: UNKNOWN only for empty values
    if agent["value"] in {None, ""}:
        agent["value"] = "UNKNOWN"
        agent["conf"] = 0.0
        agent["rule"] = "R30_EMPTY_AGENT"
        debug["needs_ml"] = True
        debug["r30_trigger"] = debug["r30_trigger"] or "empty_agent"

    if patient["value"] in {None, ""}:
        patient["value"] = "UNKNOWN"
        patient["conf"] = 0.0
        patient["rule"] = "R30_EMPTY_PATIENT"
        debug["needs_ml"] = True
        debug["r30_trigger"] = debug["r30_trigger"] or "empty_patient"

    if action["value"] in {None, ""}:
        action["value"] = "UNKNOWN_VERB"
        action["conf"] = 0.0
        action["rule"] = "R30_EMPTY_ACTION"
        debug["needs_ml"] = True
        debug["r30_trigger"] = debug["r30_trigger"] or "empty_action"

    if 0.0 < agent["conf"] < 0.70:
        debug["needs_ml"] = True
        agent["flags"]["needs_ml"] = True
        if agent["rule"] in {"R8", "R9"} and debug["ml_reason_agent"] is None:
            debug["needs_ml_agent"] = True
            debug["ml_reason_agent"] = "low_conf"

    if 0.0 < patient["conf"] < 0.70:
        debug["needs_ml"] = True
        patient["flags"]["needs_ml"] = True
        if patient["rule"] in {"R8", "R9"} and debug["ml_reason_patient"] is None:
            debug["needs_ml_patient"] = True
            debug["ml_reason_patient"] = "low_conf"

    if agent["conf"] >= 0.70:
        state.last_agent_value = agent["value"]
        state.last_agent_conf = agent["conf"]

    if patient["conf"] >= 0.70:
        state.last_patient_value = patient["value"]
        state.last_patient_conf = patient["conf"]

    # TODO: apply R2

    return {
        "action": action,
        "agent": agent,
        "patient": patient,
        "debug": debug,
    }


def parse_all_clauses(clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    state = DiscourseState()
    parsed = []
    prev_src = None
    pending_boundary_strength = None

    for clause in clauses:
        src = clause.get("source_index") or clause.get("sentence_id") or clause.get(
            "file"
        )
        if src != prev_src and src is not None:
            # Reset per source to avoid leaking pt_map/last roles across sources.
            state = DiscourseState()
            prev_src = src
        tokens = clause.get("tokens", [])
        split_meta = clause.get("split")
        boundary_seen = bool(clause.get("boundary_seen"))
        clause_flags = clause.get("clause_flags") or {}
        if boundary_seen and pending_boundary_strength:
            clause_flags = dict(clause_flags)
            clause_flags["boundary_strength"] = pending_boundary_strength
            pending_boundary_strength = None

        parsed.append(
            parse_clause(tokens, state, boundary_seen, split_meta, clause_flags)
        )
        if split_meta and split_meta.get("reason") == "R26_BOUNDARY":
            strength = (split_meta.get("flags") or {}).get("strength")
            pending_boundary_strength = strength if strength in {"strong", "weak"} else None
        else:
            pending_boundary_strength = None

    return parsed


if __name__ == "__main__":
    pass
