from typing import Optional

from token_utils import is_boundary, is_boundary_strong, is_noun, is_pt, is_verb

LIGHT_VERB_BLACKLIST = {"\u3042\u308b", "\u306a\u308b", "\u3044\u308b"}


def make_split_meta(reason: str, conf: float, at: int, flags: dict) -> dict:
    return {
        "reason": reason,
        "conf": conf,
        "at": at,
        "flags": flags,
    }


def emit_clause(
    clauses: list,
    tokens: list,
    start: int,
    end: int,
    split_meta: Optional[dict],
    boundary_seen: bool,
    clause_flags: Optional[dict] = None,
) -> int:
    if start >= end:
        return end
    clauses.append(
        {
            "tokens": tokens[start:end],
            "split": split_meta,
            "boundary_seen": boundary_seen,
            "clause_flags": (clause_flags or {}).copy(),
        }
    )
    return end


def is_verb_lexical(tok: dict) -> bool:
    if tok.get("word") in LIGHT_VERB_BLACKLIST:
        return False
    if is_verb(tok):
        return True
    modifier = tok.get("modifier") or {}
    valence = modifier.get("valence")
    return valence in {"transitive", "intransitive", "ambitransitive"}


def is_content_noun(tok: dict) -> bool:
    return is_noun(tok) and not is_verb_lexical(tok)


def explicit_patient_around_verb(tokens, i) -> tuple[bool, bool]:
    has_preverb_noun = i > 0 and is_content_noun(tokens[i - 1])
    has_pt_postverb = i + 1 < len(tokens) and is_pt(tokens[i + 1])
    return has_preverb_noun, has_pt_postverb


def segment_clauses(
    tokens: list[dict],
    split_threshold: float = 0.7,
    min_left_tokens: int = 2,
    split_decay: float = 0.05,
    max_decay_steps: int = 3,
) -> list[dict]:
    """Segment a token list into clauses.

    Returns a list of clause dicts:
      - tokens: slice of tokens
      - split: None or {reason, conf, at, flags}
      - boundary_seen: True if clause starts after a boundary
      - clause_flags: clause-level warnings (e.g., overattach_risk)
    """
    clauses = []
    start = 0
    boundary_seen = False
    verb_count = 0
    patient_consumed = False
    clause_flags = {}
    splits_so_far = 0

    for i, tok in enumerate(tokens):
        if is_boundary(tok):
            strength = "strong" if is_boundary_strong(tok) else "weak"
            split_meta = make_split_meta(
                reason="R26_BOUNDARY",
                conf=1.0,
                at=i,
                flags={
                    "token_word": tok.get("word"),
                    "strength": strength,
                },
            )
            start = emit_clause(
                clauses, tokens, start, i, split_meta, boundary_seen, clause_flags
            )
            start = i + 1
            splits_so_far += 1
            boundary_seen = True
            verb_count = 0
            patient_consumed = False
            clause_flags = {}
            continue

        if is_verb_lexical(tok):
            has_preverb_noun, has_pt_postverb = explicit_patient_around_verb(tokens, i)
            has_new_object_candidate = has_preverb_noun or has_pt_postverb

            modifier = tok.get("modifier") or {}
            valence = modifier.get("valence")

            if valence == "intransitive" and has_preverb_noun:
                clause_flags["overattach_risk"] = True
                clause_flags["overattach_verb_word"] = tok.get("word")
                clause_flags["overattach_reason"] = "intransitive_with_preverb_noun"
            if valence:
                clause_flags.setdefault("verb_valences", []).append(valence)

            candidate = None
            if verb_count >= 1:
                if patient_consumed and not has_new_object_candidate:
                    candidate = make_split_meta(
                        reason="S_PAT",
                        conf=0.75,
                        at=i,
                        flags={
                            "patient_consumed": True,
                            "has_new_object_candidate": False,
                        },
                    )
                else:
                    candidate = make_split_meta(
                        reason="S_VERB",
                        conf=0.85,
                        at=i,
                        flags={"verb_count_in_clause": verb_count + 1},
                    )

            if candidate:
                decay = split_decay * min(splits_so_far, max_decay_steps)
                candidate["conf"] = max(0.0, candidate["conf"] - decay)
                candidate["flags"]["splits_so_far"] = splits_so_far
                candidate["flags"]["conf_decay"] = decay

            if (
                candidate
                and candidate["conf"] >= split_threshold
                and (i - start) >= min_left_tokens
            ):
                split_at = i
                if (
                    i - 1 > start
                    and is_content_noun(tokens[i - 1])
                    and (i - 1 - start) >= min_left_tokens
                ):
                    split_at = i - 1
                    candidate["flags"]["shift_preverb_noun"] = True
                start = emit_clause(
                    clauses,
                    tokens,
                    start,
                    split_at,
                    candidate,
                    boundary_seen,
                    clause_flags,
                )
                start = split_at
                splits_so_far += 1
                verb_count = 0
                patient_consumed = False
                boundary_seen = False
                clause_flags = {}

            if has_new_object_candidate:
                patient_consumed = True

            verb_count += 1

    emit_clause(clauses, tokens, start, len(tokens), None, boundary_seen, clause_flags)
    return clauses
