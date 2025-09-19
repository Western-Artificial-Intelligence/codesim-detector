# token_check.py
import json, io, tokenize, keyword, math, pathlib, joblib
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

JSONL = "python_subset.jsonl"     # your file from earlier
ARTIFACTS_DIR = "artifacts"
pathlib.Path(ARTIFACTS_DIR).mkdir(exist_ok=True)

# ---------- 1) TOKEN NORMALIZER ----------
KW = set(keyword.kwlist)
def code_to_tokens(code: str):
    """
    Turn code into a token stream robust to renames/formatting:
      - keywords/operators kept
      - identifiers -> ID
      - numbers -> NUM
      - strings/docstrings -> STR
    """
    if not isinstance(code, str): 
        return []
    out = []
    try:
        # Need bytes for tokenize
        g = tokenize.tokenize(io.BytesIO(code.encode("utf-8")).readline)
        for tok in g:
            ttype, tstr = tok.type, tok.string
            if ttype == tokenize.NAME:
                out.append(tstr if tstr in KW else "ID")
            elif ttype == tokenize.NUMBER:
                out.append("NUM")
            elif ttype == tokenize.STRING:
                # collapse strings/docstrings
                out.append("STR")
            elif ttype == tokenize.OP:
                out.append(tstr)
            # ignore INDENT/DEDENT/NEWLINE/NL/ENCODING/COMMENT/etc.
    except Exception:
        # if tokenize fails, fall back: coarse mask
        out = []
    return out

def tokens_text(tokens):
    # join so TfidfVectorizer can re-gram over tokens
    return " ".join(tokens)

# ---------- 2) LOAD & HYGIENE ----------
docs = []
paths = []
with open(JSONL, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        path = ex.get("path") or "<no-path>"
        code = ex.get("code") or ""
        if not isinstance(code, str) or len(code) < 20 or len(code) > 150_000:
            continue
        toks = code_to_tokens(code)
        if len(toks) < 20: 
            continue
        docs.append(tokens_text(toks))
        paths.append(path)

print(f"Using {len(docs)} files after filtering.")

# ---------- 3) TF-IDF over token n-grams ----------
# 3–5 grams of tokens is a strong default for plagiarism
vec = TfidfVectorizer(lowercase=False, token_pattern=r"[^ ]+", ngram_range=(3,5), min_df=2)
X = vec.fit_transform(docs)

joblib.dump(vec, f"{ARTIFACTS_DIR}/token_vectorizer.joblib")
joblib.dump({"paths": paths}, f"{ARTIFACTS_DIR}/meta.joblib")
print("Vectorizer + metadata saved.")

# ---------- 4) Nearest Neighbors (cosine) ----------
nn = NearestNeighbors(metric="cosine", n_neighbors=6, n_jobs=-1)
nn.fit(X)
joblib.dump(nn, f"{ARTIFACTS_DIR}/nn_cosine.joblib")
print("NN index saved.")

# ---------- 5) Inspect top matches for each file ----------
def top_matches(ix, k=5):
    # returns (index, score) for k nearest excluding self
    dists, idxs = nn.kneighbors(X[ix], n_neighbors=k+1, return_distance=True)
    out = []
    for dist, j in zip(dists[0], idxs[0]):
        if j == ix: 
            continue
        out.append((j, 1.0 - float(dist)))  # cosine similarity
        if len(out) >= k: 
            break
    return out

# quick report of suspicious pairs (high similarity)
pairs = []
for i in range(len(paths)):
    for j, sim in top_matches(i, k=3):
        if sim >= 0.85:   # starting threshold; tune later
            a, b = sorted((i, j))
            pairs.append((a, b, sim))

# dedupe & print
seen = set()
pairs = sorted(pairs, key=lambda x: -x[2])
for a, b, sim in pairs:
    if (a,b) in seen: 
        continue
    seen.add((a,b))
    print(f"{sim:.3f} :: {paths[a]}  ~~  {paths[b]}")

# ---------- 6) Pairwise API (useful for your UI/MLP later) ----------
def token_similarity(code_a: str, code_b: str) -> float:
    ta = tokens_text(code_to_tokens(code_a))
    tb = tokens_text(code_to_tokens(code_b))
    Xa = vec.transform([ta]); Xb = vec.transform([tb])
    return float(cosine_similarity(Xa, Xb)[0,0])

# Example:
# print(token_similarity("def f(x): return x+1", "def g(y): return y + 1"))
