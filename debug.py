# dump_role_debug.py
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import token_utils
from roles_parser import DiscourseState, parse_clause


# ----------------------------
# Helpers: robust parsing
# ----------------------------
def _safe_load_py_literal(s: Any) -> Any:
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    txt = str(s).strip()
    if not txt or txt.lower() == "nan":
        return None
    try:
        return ast.literal_eval(txt)
    except Exception:
        return None


def _safe_json_loads(s: Any) -> Any:
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    txt = str(s).strip()
    if not txt or txt.lower() == "nan":
        return None
    try:
        return json.loads(txt)
    except Exception:
        return None


def _boolish(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y"}


def _get_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


# ----------------------------
# Input loader: cell segmented
# ----------------------------
def _load_rows(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Empty CSV or missing header.")

        if "tokens" not in reader.fieldnames and "tokens_json" not in reader.fieldnames:
            raise ValueError("Expected a 'tokens' or 'tokens_json' column in the input CSV.")

        for i, row in enumerate(reader):
            if limit > 0 and i >= limit:
                break
            rows.append(row)
    return rows


def _parse_tokens(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    if row.get("tokens_json"):
        data = _safe_json_loads(row["tokens_json"])
        return data if isinstance(data, list) else []
    data = _safe_load_py_literal(row.get("tokens"))
    return data if isinstance(data, list) else []


def _parse_clause_flags(row: Dict[str, Any]) -> Dict[str, Any]:
    data = _safe_load_py_literal(row.get("clause_flags"))
    return data if isinstance(data, dict) else {}


def _parse_split(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data = _safe_load_py_literal(row.get("split"))
    return data if isinstance(data, dict) else None


def _clause_id(row: Dict[str, Any], ridx: int) -> Dict[str, Any]:
    # Keep generic: if columns don't exist, they remain None
    return {
        "row_index": ridx,
        "sentence_id": row.get("sentence_id"),
        "file": row.get("file"),
        "tier": row.get("tier"),
        "clause_index": row.get("clause_index"),
        "start": row.get("start"),
        "end": row.get("end"),
    }


def _tokens_str(tokens: List[Dict[str, Any]]) -> str:
    return " ".join([str(t.get("word", "")).strip() for t in tokens]).strip()


# ----------------------------
# Main debug dump
# ----------------------------
def main(
    input_csv: str = "cell segmented.csv",
    outdir: str = "debug_dump",
    limit: int = 500,
) -> None:
    inp = Path(input_csv)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp.resolve()}")

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(inp, limit)

    state = DiscourseState()

    clause_rows: List[Dict[str, Any]] = []
    pt_rows: List[Dict[str, Any]] = []

    stats: Dict[str, Any] = {
        "n_clauses": 0,
        "n_with_verb": 0,
        "n_has_pt_anywhere": 0,
        "n_postverb_is_pt": 0,
        "n_postverb_pt_resolved": 0,
        "n_patient_R12": 0,
        "n_patient_R13": 0,
        "n_patient_R8": 0,
        "n_patient_R30": 0,
        "n_conflict_true": 0,
        "n_ruletrace_R22_ASSIGN": 0,
        "n_pt_tokens": 0,
        "n_pt_key_none": 0,
        "n_pt_key_pt": 0,
        "examples_postverb_unresolved": [],
        "examples_pt_key_none": [],
        "examples_conflict": [],
        "examples_patient_R12": [],
        "examples_patient_R13": [],
    }

    for ridx, row in enumerate(rows):
        tokens = _parse_tokens(row)
        split_meta = _parse_split(row)
        clause_flags = _parse_clause_flags(row)
        boundary_seen = _boolish(row.get("boundary_seen"))

        parsed = parse_clause(tokens, state, boundary_seen, split_meta, clause_flags)
        debug = parsed.get("debug", {}) or {}

        cid = _clause_id(row, ridx)
        tstr = _tokens_str(tokens)

        # Basic metrics
        stats["n_clauses"] += 1
        if debug.get("verb_index", -1) != -1:
            stats["n_with_verb"] += 1
        if debug.get("has_pt"):
            stats["n_has_pt_anywhere"] += 1

        postverb_is_pt = bool(debug.get("postverb_is_pt"))
        pt_resolved = bool(debug.get("pt_resolved"))

        if postverb_is_pt:
            stats["n_postverb_is_pt"] += 1
            if pt_resolved:
                stats["n_postverb_pt_resolved"] += 1
            else:
                if len(stats["examples_postverb_unresolved"]) < 30:
                    vi = _get_int(debug.get("verb_index"), -1)
                    pv_word = None
                    if vi >= 0 and vi + 1 < len(tokens):
                        pv_word = tokens[vi + 1].get("word")
                    stats["examples_postverb_unresolved"].append(
                        {**cid, "tokens_str": tstr, "postverb_word": pv_word, "postverb_pt_key": debug.get("postverb_pt_key")}
                    )

        patient_rule = parsed.get("patient", {}).get("rule")
        if patient_rule == "R12":
            stats["n_patient_R12"] += 1
            if len(stats["examples_patient_R12"]) < 30:
                stats["examples_patient_R12"].append({**cid, "tokens_str": tstr})
        elif patient_rule == "R13":
            stats["n_patient_R13"] += 1
            if len(stats["examples_patient_R13"]) < 30:
                stats["examples_patient_R13"].append({**cid, "tokens_str": tstr})
        elif isinstance(patient_rule, str) and patient_rule.startswith("R8"):
            stats["n_patient_R8"] += 1
        elif isinstance(patient_rule, str) and patient_rule.startswith("R30"):
            stats["n_patient_R30"] += 1

        if debug.get("conflict"):
            stats["n_conflict_true"] += 1
            if len(stats["examples_conflict"]) < 30:
                stats["examples_conflict"].append({**cid, "tokens_str": tstr})

        rule_trace = debug.get("rule_trace") or []
        if isinstance(rule_trace, list) and "R22_ASSIGN" in rule_trace:
            stats["n_ruletrace_R22_ASSIGN"] += 1

        # Clause row output
        clause_rows.append(
            {
                **cid,
                "tokens_str": tstr,
                "action_value": parsed.get("action", {}).get("value"),
                "action_rule": parsed.get("action", {}).get("rule"),
                "agent_value": parsed.get("agent", {}).get("value"),
                "agent_rule": parsed.get("agent", {}).get("rule"),
                "agent_conf": parsed.get("agent", {}).get("conf"),
                "patient_value": parsed.get("patient", {}).get("value"),
                "patient_rule": patient_rule,
                "patient_conf": parsed.get("patient", {}).get("conf"),
                "verb_index": debug.get("verb_index"),
                "verb_count": debug.get("verb_count"),
                "noun_count": debug.get("noun_count"),
                "has_pt": debug.get("has_pt"),
                "postverb_is_pt": postverb_is_pt,
                "postverb_pt_key": debug.get("postverb_pt_key"),
                "postverb_pt_source": debug.get("postverb_pt_source"),
                "pt_resolved": pt_resolved,
                "pt_map_size": debug.get("pt_map_size"),
                "verb_transitive": debug.get("verb_transitive"),
                "preverb_is_noun": debug.get("preverb_is_noun"),
                "conflict": debug.get("conflict"),
                "best_candidate_conf": debug.get("best_candidate_conf"),
                "rule_trace_json": json.dumps(rule_trace, ensure_ascii=False),
                "pt_resolutions_json": json.dumps(debug.get("pt_resolutions") or [], ensure_ascii=False),
                "candidates_agent_json": json.dumps(debug.get("candidates_agent") or [], ensure_ascii=False),
                "candidates_patient_json": json.dumps(debug.get("candidates_patient") or [], ensure_ascii=False),
            }
        )

        # Token-level PT output
        pt_res_map = {}
        for item in (debug.get("pt_resolutions") or []):
            idx = item.get("index")
            if isinstance(idx, int):
                pt_res_map[idx] = item

        vi = _get_int(debug.get("verb_index"), -1)

        for tidx, tok in enumerate(tokens):
            if not token_utils.is_pt(tok):
                continue

            stats["n_pt_tokens"] += 1

            pk = token_utils.pt_key(tok)
            if pk is None:
                stats["n_pt_key_none"] += 1
                if len(stats["examples_pt_key_none"]) < 30:
                    stats["examples_pt_key_none"].append(
                        {**cid, "token_index": tidx, "word": tok.get("word"), "pos": tok.get("pos"), "tokens_str": tstr}
                    )
            if pk == "pt":
                stats["n_pt_key_pt"] += 1

            is_postverb = (vi >= 0 and tidx == vi + 1)
            res = pt_res_map.get(tidx, {})

            pt_rows.append(
                {
                    **cid,
                    "token_index": tidx,
                    "is_postverb": is_postverb,
                    "word": tok.get("word"),
                    "pos": tok.get("pos"),
                    "lemma": tok.get("lemma"),
                    "mods_raw_json": json.dumps(tok.get("mods_raw") or [], ensure_ascii=False),
                    "modifier_json": json.dumps(tok.get("modifier") or {}, ensure_ascii=False),
                    "pt_key_current": pk,
                    "resolved_value": res.get("resolved_value"),
                    "resolved_conf": res.get("resolved_conf"),
                    "resolved_source": res.get("source"),
                    "debug_pt_key_logged": res.get("pt_key"),
                    "debug_raw_word_logged": res.get("raw_word"),
                }
            )

    # Write outputs
    clauses_path = out / "debug_clauses.csv"
    pt_path = out / "debug_pt_tokens.csv"
    summary_path = out / "debug_summary.txt"

    if clause_rows:
        with clauses_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(clause_rows[0].keys()))
            w.writeheader()
            w.writerows(clause_rows)

    if pt_rows:
        with pt_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pt_rows[0].keys()))
            w.writeheader()
            w.writerows(pt_rows)

    # Summary
    lines: List[str] = []
    lines.append("=== ROLE PARSER DEBUG SUMMARY ===")
    lines.append(f"input: {inp.resolve()}")
    lines.append(f"limit: {limit}")
    lines.append("")

    for k, v in stats.items():
        if k.startswith("examples_"):
            continue
        lines.append(f"{k}: {v}")

    def dump_examples(title: str, key: str) -> None:
        ex = stats.get(key) or []
        lines.append("")
        lines.append(f"--- {title} (showing {len(ex)} examples) ---")
        for e in ex:
            lines.append(json.dumps(e, ensure_ascii=False))

    dump_examples("POSTVERB PT UNRESOLVED", "examples_postverb_unresolved")
    dump_examples("PT KEY NONE", "examples_pt_key_none")
    dump_examples("CONFLICT TRUE", "examples_conflict")
    dump_examples("PATIENT R12", "examples_patient_R12")
    dump_examples("PATIENT R13", "examples_patient_R13")

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:")
    print(" -", clauses_path.resolve())
    print(" -", pt_path.resolve())
    print(" -", summary_path.resolve())


if __name__ == "__main__":
    # Defaults: edit here if you want
    main(input_csv="cell segmented.csv", outdir="debug_dump", limit=500)
