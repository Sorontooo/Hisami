import json
import os
import sys

import requests
import numpy as np
import hnswlib
from tqdm import tqdm

from memory.config import load_config as _load_config, ollama_base

# ============================================================
# CONFIG
# ============================================================

try:
    OLLAMA_URL = ollama_base(_load_config().get("ollama_url"))
except Exception:
    OLLAMA_URL = ollama_base(None)
EMBED_MODEL = "nomic-embed-text"

INPUT_FILE = "dataset.jsonl"
INDEX_FILE = "index.bin"   # changed extension (optional)
TEXTS_FILE = "texts.json"

DIM = 768  # nomic embedding size

# Refuse to overwrite a good index with a partial one: abort (writing
# nothing) when more than this share of records fail to embed, or after
# this many failures in a row (Ollama went away mid-run).
MAX_SKIP_RATIO = 0.10
MAX_CONSECUTIVE_FAILURES = 25
REQUEST_TIMEOUT = 60

# HNSW params (safe defaults)
M = 16
EF_CONSTRUCTION = 200

# ============================================================
# EMBEDDING
# ============================================================

def get_embedding(text):
    res = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": text
        },
        timeout=REQUEST_TIMEOUT,
    )
    res.raise_for_status()
    return res.json()["embeddings"][0]

# ============================================================
# BUILD TEXT (WITH WEIGHTING)
# ============================================================

def build_text(record):

    context = "\n".join(
        f"{m['author']}: {m['content']}"
        for m in record.get("context", [])
    )

    if record.get("type") == "reply":
        prefix = "HIGH VALUE INTERACTION (conversation reply, strong behavioral signal)"
    else:
        prefix = "STYLE EXAMPLE (tone, phrasing, personality)"

    input_text = record.get("input")
    if input_text == "reply_detected":
        input_text = "implicit reply in conversation"

    return f"""
{prefix}

Context:
{context}

User Input:
{input_text}

Target Response:
{record.get("response")}
""".strip()

# ============================================================
# MAIN
# ============================================================

def main():

    texts = []
    embeddings = []

    total = 0
    skipped = 0
    consecutive = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Building index"):
            if not line.strip():
                continue
            record = json.loads(line)

            text = build_text(record)

            try:
                emb = get_embedding(text)

                embeddings.append(emb)
                texts.append(text)
                total += 1
                consecutive = 0

            except Exception as e:
                print(f"Error embedding: {e}")
                skipped += 1
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    sys.exit(f"Aborting: {consecutive} embeddings failed in a "
                             f"row (is Ollama running at {OLLAMA_URL}?). "
                             f"Existing {INDEX_FILE}/{TEXTS_FILE} untouched.")
                continue

    if not embeddings:
        sys.exit(f"Aborting: nothing embedded ({skipped} failed). "
                 f"Existing {INDEX_FILE}/{TEXTS_FILE} untouched.")
    if skipped > MAX_SKIP_RATIO * (total + skipped):
        sys.exit(f"Aborting: {skipped}/{total + skipped} embeddings failed "
                 f"(> {MAX_SKIP_RATIO:.0%}). Fix the errors above and rerun; "
                 f"existing {INDEX_FILE}/{TEXTS_FILE} untouched.")

    # ===== BUILD INDEX =====
    print("Initializing HNSW index...")

    index = hnswlib.Index(space='l2', dim=DIM)
    index.init_index(
        max_elements=len(embeddings),
        ef_construction=EF_CONSTRUCTION,
        M=M
    )

    index.add_items(
        np.array(embeddings).astype("float32"),
        np.arange(len(embeddings))
    )

    # optional but recommended for search quality
    index.set_ef(50)

    # ===== SAVE =====
    # Write to temp files, then swap both in, so a crash mid-save never
    # leaves a truncated index or an index/texts mismatch.
    index.save_index(INDEX_FILE + ".tmp")

    with open(TEXTS_FILE + ".tmp", "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    os.replace(INDEX_FILE + ".tmp", INDEX_FILE)
    os.replace(TEXTS_FILE + ".tmp", TEXTS_FILE)

    print("\n==============================")
    print(f"Indexed: {total}")
    print(f"Skipped: {skipped}")
    print("==============================")

# ============================================================

if __name__ == "__main__":
    main()
