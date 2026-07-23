import ast
import csv

from segmentation import segment_clauses


def load_phrases(path: str, limit: int | None = None):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tokens_raw = row.get("tokens")
            pos_raw = row.get("pos")
            if not tokens_raw:
                continue
            try:
                words = ast.literal_eval(tokens_raw)
            except Exception:
                continue
            pos_list = []
            if pos_raw:
                try:
                    pos_list = ast.literal_eval(pos_raw)
                except Exception:
                    pos_list = []
            rows.append((row, words, pos_list))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_zrz_maps(path: str):
    by_id = {}
    by_range = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tokens_raw = row.get("tokens")
            if not tokens_raw:
                continue
            try:
                tokens = ast.literal_eval(tokens_raw)
            except Exception:
                continue
            file_name = row.get("file")
            sentence_id = str(row.get("sentence_id"))
            start = row.get("start")
            end = row.get("end")
            key_id = (file_name, sentence_id)
            if key_id not in by_id:
                by_id[key_id] = tokens
            try:
                key_range = (file_name, int(start), int(end))
            except Exception:
                key_range = None
            if key_range and key_range not in by_range:
                by_range[key_range] = tokens
    return by_id, by_range


def load_jsl_tokens_by_file(path: str):
    by_file = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row.get("file")
            if not file_name:
                continue
            try:
                start = int(row.get("start"))
                end = int(row.get("end"))
            except Exception:
                continue
            word = row.get("word")
            pos = row.get("pos")
            by_file.setdefault(file_name, []).append((start, end, word, pos))
    return by_file


def build_token_dicts(words: list, pos_list: list) -> list[dict]:
    tokens = []
    for i, word in enumerate(words):
        pos = pos_list[i] if i < len(pos_list) else None
        tokens.append({"word": word, "pos": pos})
    return tokens


def segmented_phrase(clauses: list[dict]) -> str:
    parts = []
    for clause in clauses:
        words = [t.get("word", "") for t in clause["tokens"]]
        parts.append(" ".join(words))
    return " | ".join(parts)


def split_info(clauses: list[dict]) -> str:
    parts = []
    for clause in clauses:
        split = clause.get("split")
        if not split:
            continue
        reason = split.get("reason")
        conf = split.get("conf")
        at = split.get("at")
        flags = split.get("flags")
        parts.append(f"{reason}@{at} conf={conf} flags={flags}")
    return " ; ".join(parts)

def collect_jsl_tokens(by_file: dict, file_name: str, start: int, end: int) -> str:
    items = by_file.get(file_name, [])
    words = []
    for s, e, word, _pos in items:
        if s >= start and e <= end:
            words.append(word)
    return " ".join([w for w in words if w])


def main():
    rows = load_phrases("data/jls_phrases.csv", limit=None)
    zrz_by_id, zrz_by_range = load_zrz_maps("data/zrz.csv")
    jsl_tokens_by_file = load_jsl_tokens_by_file("data/jsl_tokens_train.csv")

    out_path = "data/segmentation_comparison.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sentence_id",
                "file",
                "start",
                "end",
                "phrase_source",
                "phrase_segmented",
                "split_info",
                "jsl_tokens_train",
            ]
        )
        for row, words, pos_list in rows:
            file_name = row.get("file")
            sentence_id = str(row.get("sentence_id", "?"))
            try:
                start = int(row.get("start"))
                end = int(row.get("end"))
            except Exception:
                start = -1
                end = -1

            zrz_tokens = zrz_by_id.get((file_name, sentence_id))
            if zrz_tokens is None:
                zrz_tokens = zrz_by_range.get((file_name, start, end))

            if zrz_tokens is None:
                tokens = build_token_dicts(words, pos_list)
            else:
                tokens = zrz_tokens

            clauses = segment_clauses(tokens, split_threshold=0.7)
            source = " ".join(words)
            segmented = segmented_phrase(clauses)
            split_details = split_info(clauses)
            jsl_tokens = ""
            if start >= 0 and end >= 0:
                jsl_tokens = collect_jsl_tokens(
                    jsl_tokens_by_file, file_name, start, end
                )

            writer.writerow(
                [
                    sentence_id,
                    file_name,
                    start,
                    end,
                    source,
                    segmented,
                    split_details,
                    jsl_tokens,
                ]
            )

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

