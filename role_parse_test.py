from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import token_utils
from roles_parser import DiscourseState, parse_clause
from segmentation import segment_clauses


def _verb_token(word: str = "eat") -> Dict[str, Any]:
    return {"word": word, "pos": "verb", "modifier": {"valence": "transitive"}}


def test_pt_bare_postverb_unresolved() -> None:
    tokens = [_verb_token("みる"), {"word": "pt", "pos": "代名詞"}]
    parsed = parse_clause(tokens, DiscourseState(), boundary_seen=False)
    debug = parsed["debug"]
    patient = parsed["patient"]

    assert patient["rule"] == "R13"
    assert debug.get("r13_reason") == "PT_BARE"
    assert not debug.get("block_patient_fill", False)
    pt_resolutions = debug.get("pt_resolutions") or []
    assert any(entry.get("source") == "UNRESOLVED" for entry in pt_resolutions)


def test_pt_inline_word_resolves_patient() -> None:
    tokens = [_verb_token("買う"), {"word": "pt(女)", "pos": "代名詞"}]
    parsed = parse_clause(tokens, DiscourseState(), boundary_seen=False)

    assert parsed["patient"]["value"] == "女"
    assert parsed["patient"]["rule"] == "R12"
    assert parsed["debug"].get("postverb_pt_source") == "PT_INLINE"
    assert parsed["debug"].get("needs_ml_patient") is False


def test_pt_inline_word_resolves_patient_with_colon() -> None:
    tokens = [_verb_token(), {"word": "pt:カレー", "pos": "代名詞"}]
    parsed = parse_clause(tokens, DiscourseState(), boundary_seen=False)

    assert parsed["patient"]["value"] == "カレー"
    assert parsed["patient"]["rule"] == "R12"
    assert parsed["debug"].get("postverb_pt_source") == "PT_INLINE"


def test_pt_inline_nested_content_resolves() -> None:
    tokens = [
        {"word": "猫", "pos": "名詞"},
        _verb_token("外す"),
        {"word": "pt(左手:リスト)", "pos": "代名詞"},
    ]
    parsed = parse_clause(tokens, DiscourseState(), boundary_seen=False)

    assert parsed["patient"]["value"] == "左手:リスト"
    assert parsed["patient"]["rule"] == "R12"
    assert parsed["debug"].get("postverb_pt_source") == "PT_INLINE"


def test_self_ref_drops_agent_from_r8() -> None:
    state = DiscourseState(topic_value="猫", topic_conf=0.70)
    tokens = [_verb_token("見る"), {"word": "猫", "pos": "名詞"}]
    parsed = parse_clause(tokens, state, boundary_seen=False)

    assert parsed["agent"]["value"] == "UNKNOWN"
    assert parsed["agent"]["rule"] == "R23_DROP_AGENT_R8"
    assert parsed["debug"].get("self_ref_repaired") == "drop_agent_r8"
    assert parsed["patient"]["value"] == "猫"
    assert parsed["patient"]["rule"] == "R10"


def test_r9_agent_needs_ml() -> None:
    tokens = [_verb_token("見る")]
    parsed = parse_clause(tokens, DiscourseState(), boundary_seen=False)

    assert parsed["agent"]["value"] == "SELF"
    assert parsed["agent"]["rule"] == "R9"
    assert parsed["debug"].get("needs_ml_agent") is True


def test_r8_patient_needs_ml_with_low_conf() -> None:
    state = DiscourseState(last_patient_value="箱", last_patient_conf=0.55)
    tokens = [_verb_token("見る")]
    parsed = parse_clause(tokens, state, boundary_seen=False)

    assert parsed["patient"]["rule"] == "R8"
    assert parsed["patient"]["value"] == "箱"
    assert parsed["debug"].get("needs_ml_patient") is True
    assert parsed["debug"].get("ml_reason_patient") == "low_conf"
    assert len(parsed["debug"].get("patient_candidates") or []) == 4


def test_r8_patient_low_conf_no_r30() -> None:
    state = DiscourseState(last_patient_value="箱", last_patient_conf=0.20)
    tokens = [_verb_token("見る")]
    parsed = parse_clause(tokens, state, boundary_seen=False)

    assert parsed["patient"]["rule"] == "R8"
    assert parsed["patient"]["value"] == "箱"
    assert parsed["debug"].get("needs_ml_patient") is True
    assert parsed["debug"].get("ml_reason_patient") == "low_conf"
    assert parsed["patient"]["value"] != "UNKNOWN"
    assert parsed["patient"]["rule"] != "R30_EMPTY_PATIENT"


def test_r30_empty_patient_triggers() -> None:
    tokens = [_verb_token("見る"), {"word": "", "pos": "名詞"}]
    parsed = parse_clause(tokens, DiscourseState(), boundary_seen=False)

    assert parsed["patient"]["value"] == "UNKNOWN"
    assert parsed["patient"]["rule"] == "R30_EMPTY_PATIENT"
    assert parsed["debug"].get("r30_trigger") == "empty_patient"


def test_boundary_strength_decay_and_reset() -> None:
    state = DiscourseState(last_patient_value="箱", last_patient_conf=0.8)
    parse_clause([], state, boundary_seen=True, clause_flags={"boundary_strength": "weak"})
    assert abs(state.last_patient_conf - 0.4) < 1e-6
    assert state.last_patient_value == "箱"

    parse_clause([], state, boundary_seen=True, clause_flags={"boundary_strength": "strong"})
    assert state.last_patient_conf == 0.0
    assert state.last_patient_value is None
def test_pt_key_normalization() -> None:
    assert token_utils.pt_key({"word": "pt２"}) == "pt2"
    assert token_utils.pt_key({"word": "pt2（]）"}) == "pt2"


def _load_phrases(path: Path, limit: int | None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "tokens_json" not in reader.fieldnames:
            raise ValueError("tokens_json column not found in input CSV")
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _build_clause_rows(
    rows: List[Dict[str, Any]], clause_limit: int | None
) -> List[Dict[str, Any]]:
    clauses_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        tokens = json.loads(row["tokens_json"])
        clauses = segment_clauses(tokens)
        for clause_idx, clause in enumerate(clauses):
            clauses_rows.append(
                {
                    "source_index": idx,
                    "sentence_id": row.get("sentence_id"),
                    "file": row.get("file"),
                    "clause_index": clause_idx,
                    "tokens": clause["tokens"],
                    "split": clause["split"],
                    "boundary_seen": clause["boundary_seen"],
                    "clause_flags": clause.get("clause_flags"),
                }
            )
            if clause_limit is not None and len(clauses_rows) >= clause_limit:
                return clauses_rows
    return clauses_rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "source_index",
        "clause_index",
        "source_id",
        "tokens_str",
        "tokens_json",
        "verb_word",
        "preverb_word",
        "postverb_word",
        "postverb_pt_key_raw",
        "action_value",
        "action_conf",
        "action_rule",
        "agent_value",
        "agent_conf",
        "agent_rule",
        "patient_value",
        "patient_conf",
        "patient_rule",
        "debug_json",
        "state_before_json",
        "candidates_agent_json",
        "candidates_patient_json",
        "pt_resolutions_json",
        "needs_ml",
        "self_ref_repaired",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            debug = {}
            debug_json = row.get("debug_json")
            if debug_json:
                try:
                    debug = json.loads(debug_json)
                except json.JSONDecodeError:
                    debug = {}
            writer.writerow(
                {
                    "source_index": row.get("source_index", ""),
                    "clause_index": row.get("clause_index", ""),
                    "source_id": row.get("source_id", ""),
                    "tokens_str": row.get("tokens_str", ""),
                    "tokens_json": row.get("tokens_json", ""),
                    "verb_word": row.get("verb_word", ""),
                    "preverb_word": row.get("preverb_word", ""),
                    "postverb_word": row.get("postverb_word", ""),
                    "postverb_pt_key_raw": row.get("postverb_pt_key_raw", ""),
                    "action_value": row["action"]["value"],
                    "action_conf": row["action"]["conf"],
                    "action_rule": row["action"]["rule"],
                    "agent_value": row["agent"]["value"],
                    "agent_conf": row["agent"]["conf"],
                    "agent_rule": row["agent"]["rule"],
                    "patient_value": row["patient"]["value"],
                    "patient_conf": row["patient"]["conf"],
                    "patient_rule": row["patient"]["rule"],
                    "debug_json": row.get("debug_json", ""),
                    "state_before_json": row.get("state_before_json", ""),
                    "candidates_agent_json": row.get("candidates_agent_json", ""),
                    "candidates_patient_json": row.get(
                        "candidates_patient_json", ""
                    ),
                    "pt_resolutions_json": row.get("pt_resolutions_json", ""),
                    "needs_ml": debug.get("needs_ml", ""),
                    "self_ref_repaired": debug.get("self_ref_repaired", ""),
                }
            )


def _first_source_id(clause_row: Dict[str, Any]) -> Any:
    for key in ["source_index", "sentence_id", "file"]:
        value = clause_row.get(key)
        if value is not None and value != "":
            return value
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Role parse test on clauses JSON.")
    parser.add_argument(
        "--input",
        default="data/test2.csv",
        help="CSV with tokens_json column",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSONL output path",
    )
    parser.add_argument(
        "--output-csv",
        default="data/role_parse_output.csv",
        help="CSV output path",
    )
    parser.add_argument(
        "--output-debug-csv",
        default="data/role_parse_output_debug.csv",
        help="Debug CSV output path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of input rows",
    )
    parser.add_argument(
        "--clause-limit",
        type=int,
        default=200,
        help="Limit number of clauses processed",
    )
    parser.add_argument(
        "--print",
        type=int,
        default=0,
        dest="print_count",
        help="Print first N parsed clauses",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    output_csv = Path(args.output_csv) if args.output_csv else None
    output_debug_csv = Path(args.output_debug_csv) if args.output_debug_csv else None

    rows = _load_phrases(input_path, args.limit)
    clause_rows = _build_clause_rows(rows, args.clause_limit)

    state = DiscourseState()
    prev_src = None
    parsed = []
    for clause in clause_rows:
        src = clause.get("source_index") or clause.get("sentence_id") or clause.get(
            "file"
        )
        if src != prev_src and src is not None:
            state = DiscourseState()
            prev_src = src

        state_before = {
            "topic_value": state.topic_value,
            "topic_conf": state.topic_conf,
            "last_agent_value": state.last_agent_value,
            "last_agent_conf": state.last_agent_conf,
            "last_patient_value": state.last_patient_value,
            "last_patient_conf": state.last_patient_conf,
            "pt_map_keys": list(state.pt_map.keys()),
            "pt_map_size": len(state.pt_map),
        }

        parsed.append(
            parse_clause(
                clause.get("tokens", []),
                state,
                bool(clause.get("boundary_seen")),
                clause.get("split"),
                clause.get("clause_flags") or {},
            )
        )
        clause["state_before"] = state_before

    out_rows = []
    for clause_row, parsed_row in zip(clause_rows, parsed):
        tokens = clause_row["tokens"]
        verb_index = parsed_row.get("debug", {}).get("verb_index", -1)
        verb_word = ""
        preverb_word = ""
        postverb_word = ""
        postverb_pt_key_raw = ""
        if verb_index is not None and verb_index >= 0 and verb_index < len(tokens):
            verb_word = tokens[verb_index].get("word", "")
        if verb_index is not None and verb_index >= 1:
            preverb_word = tokens[verb_index - 1].get("word", "")
        if verb_index is not None and verb_index >= 0 and verb_index + 1 < len(tokens):
            tok_post = tokens[verb_index + 1]
            postverb_word = tok_post.get("word", "")
            if token_utils.is_pt(tok_post):
                postverb_pt_key_raw = tok_post.get("word", "")

        debug = parsed_row.get("debug", {})
        candidates_agent = debug.get("candidates_agent")
        candidates_patient = debug.get("candidates_patient")
        pt_resolutions = debug.get("pt_resolutions")

        state_before = clause_row.get("state_before")
        state_before_json = (
            json.dumps(state_before, ensure_ascii=False) if state_before else ""
        )

        out_rows.append(
            {
                "source_index": clause_row.get("source_index", ""),
                "clause_index": clause_row.get("clause_index", ""),
                "source_id": _first_source_id(clause_row),
                "tokens_str": " ".join([t.get("word", "") for t in tokens]).strip(),
                "tokens_json": json.dumps(tokens, ensure_ascii=False),
                "verb_word": verb_word,
                "preverb_word": preverb_word,
                "postverb_word": postverb_word,
                "postverb_pt_key_raw": postverb_pt_key_raw,
                "state_before_json": state_before_json,
                "candidates_agent_json": json.dumps(candidates_agent, ensure_ascii=False)
                if candidates_agent is not None
                else "",
                "candidates_patient_json": json.dumps(
                    candidates_patient, ensure_ascii=False
                )
                if candidates_patient is not None
                else "",
                "pt_resolutions_json": json.dumps(pt_resolutions, ensure_ascii=False)
                if pt_resolutions is not None
                else "",
                "action": parsed_row["action"],
                "agent": parsed_row["agent"],
                "patient": parsed_row["patient"],
                "debug_json": json.dumps(debug, ensure_ascii=False),
            }
        )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_path, out_rows)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(output_csv, out_rows)
    if output_debug_csv:
        output_debug_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(output_debug_csv, out_rows)

    if args.print_count > 0:
        for row in out_rows[: args.print_count]:
            print(
                json.dumps(
                    {
                        "source_index": row["source_index"],
                        "clause_index": row["clause_index"],
                        "action": row["action"],
                        "agent": row["agent"],
                        "patient": row["patient"],
                    },
                    ensure_ascii=False,
                )
            )

    if output_path:
        print(f"Wrote {len(out_rows)} rows to {output_path}")
    if output_csv:
        print(f"Wrote {len(out_rows)} rows to {output_csv}")
    if output_debug_csv:
        print(f"Wrote {len(out_rows)} rows to {output_debug_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
