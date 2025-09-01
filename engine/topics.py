import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


_STOP_DE = {
    "und","oder","aber","doch","nicht","kein","keine","der","die","das","den","dem","des",
    "ein","eine","einen","einem","eines","ich","du","er","sie","es","wir","ihr","ihnen","mich","dich",
    "mir","dir","sich","am","im","in","auf","an","zu","mit","von","für","aus","als","bei","so","wie",
    "auch","dass","da","weil","wenn","war","ist","sind","sein","bin","bist","wird","werden","wurden",
    "mal","noch","schon","nur","sehr","hier","dort","dann","doch","eben","zwar","etwas","mehr","weniger",
}
_STOP_EN = {
    "and","or","but","not","no","the","a","an","i","you","he","she","it","we","they","me","him","her",
    "my","your","our","their","is","are","was","were","be","been","am","in","on","at","to","for","of",
    "from","as","by","so","that","this","these","those","with","about","just","very","more","less","here","there",
}


def _tokens(text: str) -> List[str]:
    s = (text or "").lower()
    # keep unicode word chars
    toks = re.findall(r"\w+", s, flags=re.UNICODE)
    return [t for t in toks if len(t) >= 3]


def _stop_filtered(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        if t in _STOP_DE or t in _STOP_EN:
            continue
        out.append(t)
    return out


def _idf(df: Dict[str, int], n_docs: int) -> Dict[str, float]:
    import math
    return {t: math.log(max(1.0, n_docs) / max(1, df.get(t, 0))) for t in df}


def extract_topics(segments: List[Dict], max_topics: int = 8, min_weight: float = 0.0) -> List[Dict]:
    """
    Lightweight topic hints from TF-IDF aggregated across segments.
    Returns list of {label, weight, keywords} sorted by weight.
    """
    docs: List[List[str]] = []
    for s in segments:
        toks = _stop_filtered(_tokens(str(s.get("text", ""))))
        docs.append(toks)
    if not docs:
        return []

    # document frequency
    df: Dict[str, int] = defaultdict(int)
    for toks in docs:
        for t in set(toks):
            df[t] += 1
    idf = _idf(df, len(docs))
    # aggregate tf-idf
    scores: Dict[str, float] = defaultdict(float)
    for toks in docs:
        tf = Counter(toks)
        for t, c in tf.items():
            scores[t] += float(c) * idf.get(t, 0.0)
    # select top tokens as topic labels, group by simple stems (heuristic)
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    topics: List[Dict] = []
    used: set = set()
    for t, w in top:
        if len(topics) >= max_topics:
            break
        if w < min_weight:
            break
        # avoid near-duplicates by prefix check
        if any(t.startswith(u) or u.startswith(t) for u in used):
            continue
        used.add(t)
        topics.append({"label": t, "weight": round(float(w), 3), "keywords": [t]})
    return topics

