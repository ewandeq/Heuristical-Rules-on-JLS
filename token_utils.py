import re
import unicodedata


def _norm_str(value):
    return str(value).strip().lower() if value is not None else ""


def norm_nfkc(value):
    return unicodedata.normalize("NFKC", str(value)).strip().lower() if value is not None else ""


def pt_inline_referent(word):
    norm = norm_nfkc(word)
    match = re.match(r"^pt[:：](.+)$", norm)
    if not match:
        match = re.match(r"^pt[（(](.+)[)）]$", norm)
    if not match:
        return None
    referent = match.group(1).strip()
    return referent or None


def is_unknown(t):
    w = _norm_str(t.get("word"))
    p = _norm_str(t.get("pos"))
    return w in {"unknown", "?"} or p in {"unknown", "?"}


def is_verb(t):
    p = _norm_str(t.get("pos"))
    return p in {"verb", "v", "動詞", "†<Š¸z"} or p.startswith("v")


def is_verb_token(word, pos, valence_dict, blacklist=None):
    p = _norm_str(pos)
    if "動詞" in str(pos) or p in {"verb", "v", "†<Š¸z"} or p.startswith("v"):
        return True
    if word in (valence_dict or {}):
        return True
    if word and str(word).endswith("る"):
        blacklist = blacklist or {
            "メール",
            "ボール",
            "ルール",
            "タオル",
            "ビール",
            "ホール",
            "ガール",
            "ロール",
            "コール",
            "セール",
            "カール",
            "テール",
            "ファイル",
            "スタイル",
            "コイル",
            "ドリル",
            "シール",
            "ソウル",
            "オイル",
            "モール",
        }
        return word not in blacklist
    return False


def is_noun(t):
    p = _norm_str(t.get("pos"))
    return p in {"noun", "n", "名詞", "†??Š¸z"} or p.startswith("n")


def is_adj(t):
    p = _norm_str(t.get("pos"))
    return p in {"adj", "adjective", "形容詞", "形容動詞", "†«½†©ûŠ¸z", "†«½†©û†<Š¸z"} or p.startswith("adj")


def is_boundary_strong(t):
    modifier = t.get("modifier") or {}
    boundary = _norm_str(modifier.get("boundary"))
    return boundary in {"strong", "1", "true"}


def is_boundary_weak(t):
    modifier = t.get("modifier") or {}
    boundary = _norm_str(modifier.get("boundary"))
    return boundary == "weak"


def is_boundary(t):
    return is_boundary_strong(t) or is_boundary_weak(t)


def is_pt(t):
    w = _norm_str(t.get("word"))
    return w.startswith("pt")


def pt_key(t):
    if not is_pt(t):
        return None
    word = t.get("word")
    if word is None:
        return None
    norm = norm_nfkc(word)
    if norm == "pt":
        return "pt"
    if re.match(r"^pt\d+$", norm):
        return norm
    match = re.match(r"^(pt\d+)", norm)
    if match:
        return match.group(1)
    if (
        norm.startswith("pt:")
        or norm.startswith("pt：")
        or norm.startswith("pt(")
        or norm.startswith("pt（")
    ):
        return "pt"
    return None


def classify_verb_valence(lemma):
    try:
        from jamdict import Jamdict
    except Exception:
        return "unknown"
    jd = Jamdict()
    try:
        res = jd.lookup(lemma)
    except Exception:
        return "unknown"
    pos_tags = set()
    for entry in getattr(res, "entries", []) or []:
        for sense in getattr(entry, "senses", []) or []:
            for tag in getattr(sense, "pos", []) or []:
                pos_tags.add(_norm_str(tag))
    if not pos_tags:
        return "unknown"
    has_t = False
    has_i = False
    for tag in pos_tags:
        if "intransitive" in tag or tag == "vi" or tag.endswith("vi"):
            has_i = True
            continue
        if "transitive" in tag or tag == "vt" or tag.endswith("vt"):
            has_t = True
            continue
    if has_t and has_i:
        return "ambitransitive"
    if has_t:
        return "transitive"
    if has_i:
        return "intransitive"
    return "unknown"


def apply_R29_merge_unknown_runs(tokens):
    merged = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not is_unknown(t):
            merged.append(t)
            i += 1
            continue
        run_len = 1
        j = i + 1
        while j < len(tokens) and is_unknown(tokens[j]):
            run_len += 1
            j += 1
        base = dict(t)
        base["word"] = "UNKNOWN"
        base_pos = _norm_str(base.get("pos"))
        base["pos"] = base.get("pos") if base_pos == "unknown" else "unknown"
        modifier = base.get("modifier") or {}
        base["modifier"] = modifier
        base["modifier"]["sequenced"] = run_len
        merged.append(base)
        i = j
    return merged


def apply_R24R7_merge_repetitions(tokens):
    merged = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        key = _norm_str(t.get("lemma") or t.get("word"))
        run_len = 1
        j = i + 1
        while j < len(tokens):
            tj = tokens[j]
            key_j = _norm_str(tj.get("lemma") or tj.get("word"))
            if key_j != key:
                break
            run_len += 1
            j += 1
        if run_len == 1:
            merged.append(t)
            i += 1
            continue
        base = dict(t)
        modifier = base.get("modifier") or {}
        base["modifier"] = modifier
        word_norm = _norm_str(base.get("word"))
        if is_pt(base):
            base["modifier"]["pt_emphasis"] = run_len
        elif word_norm in {"終わる", "終わり", "owari"}:
            base["modifier"]["boundary_emphasis"] = run_len
        elif is_verb(base):
            base["modifier"]["iterative"] = run_len
        else:
            base["modifier"]["iterative"] = run_len
        merged.append(base)
        i = j
    return merged


def apply_rule_to_corpus(tokens_list, rule_fn):
    return [rule_fn(toks) for toks in tokens_list]
