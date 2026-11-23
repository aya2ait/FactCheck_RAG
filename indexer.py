
import json, os
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import config

def _iter_batches(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def build_index():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    print("Loading Wikipedia corpus ...")
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=False
    )
    if config.WIKI_MAX_ROWS:
        ds = ds.select(range(min(config.WIKI_MAX_ROWS, len(ds))))

    # chunk each article into ~200 word passages
    passages = []
    pid = 0
    for row in ds:
        words = row["text"].split()
        for i in range(0, len(words), 200):
            chunk = " ".join(words[i : i + 200])
            if len(chunk.strip()) > 30:
                passages.append({"id": pid, "title": row["title"], "text": chunk})
                pid += 1
        if pid >= config.WIKI_MAX_ROWS:
            break

    print(f"  {len(passages):,} passages created.")

    print(f"Embedding with {config.EMBED_MODEL} ...")
    model = SentenceTransformer(config.EMBED_MODEL)
    texts = [p["text"] for p in passages]
    all_embeddings = []

    for batch in tqdm(_iter_batches(texts, config.EMBED_BATCH), desc="Batches"):
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(vecs)

    embeddings = np.vstack(all_embeddings).astype("float32")
    print(f"  Embedding matrix: {embeddings.shape}")

    print("Building FAISS index ...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, config.INDEX_PATH)
    print(f"  Saved to {config.INDEX_PATH} ({index.ntotal:,} vectors)")

    with open(config.PASSAGES_PATH, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  Saved to {config.PASSAGES_PATH}")

build_index()
