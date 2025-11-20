#!/usr/bin/env python3
"""
CLI runner for Mini RAG System
Usage:
  python main.py --csv data/wiki_movie_plots_deduped.csv --query "Which movie..." [--mock]
""" 
import os, json, argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from termcolor import colored
try:
    from google import genai
except Exception:
    genai = None
CHUNK_SIZE_WORDS = 300
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_TOP_K = 3
def cprint(msg, color="cyan"):
    try:
        print(colored(msg, color))
    except Exception:
        print(msg)
def load_data(path, max_rows=400):
    df = pd.read_csv(path).head(max_rows)
    df = df.loc[:, ["Title","Plot"]].dropna().reset_index(drop=True)
    cprint(f"Loaded {len(df)} rows from {path}", "green")
    return df
def chunk_text(text, chunk_size_words=CHUNK_SIZE_WORDS):
    words = text.split()
    return [" ".join(words[i:i+chunk_size_words]).strip() for i in range(0, len(words), chunk_size_words) if words[i:i+chunk_size_words]]
def build_chunks(df):
    chunks = []; meta = []
    for _, row in df.iterrows():
        for ch in chunk_text(row["Plot"]):
            chunks.append(ch); meta.append({"title": row["Title"], "chunk": ch})
    cprint(f"Built {len(chunks)} chunks", "green")
    return chunks, meta
def embed_chunks(model, chunks):
    emb = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    emb = np.array(emb, dtype="float32")
    cprint(f"Embeddings shape: {emb.shape}", "green")
    return emb
def build_faiss_index(emb):
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    cprint(f"FAISS index built. N={index.ntotal}", "green")
    return index
def retrieve(query, model, index, meta, top_k=FAISS_TOP_K):
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, top_k)
    hits = []
    for idx in indices[0]:
        if idx < 0: continue
        hits.append(meta[idx])
    return hits
def get_genai_client(mock):
    if mock:
        return None
    if genai is None:
        raise RuntimeError("google-genai not installed. Use --mock to avoid API calls.")
    key = os.environ.get("GOOGLE_API_KEY", None)
    if not key:
        raise RuntimeError("GOOGLE_API_KEY missing. Export it or use --mock.")
    return genai.Client(api_key=key)
def generate_answer(query, contexts, mock=False):
    ctx_text = "\\n\\n".join([f"{c['title']}: {c['chunk']}" for c in contexts])
    prompt = (
        "You are a movie RAG assistant. Answer the question using ONLY the retrieved plot snippets. "
        "Return ONLY JSON with fields: answer, contexts, reasoning.\\n\\n"
        "Question:\\n" + query + "\\n\\nRetrieved Context:\\n" + ctx_text + "\\n\\nOutput JSON only."
    )
    if mock:
        return json.dumps({
            "answer": "Mock answer (no API). Check contexts for likely movies.",
            "contexts": [c["chunk"] for c in contexts],
            "reasoning": "Mock mode - deterministic demo."
        }, ensure_ascii=False)
    client = get_genai_client(mock)
    try:
        resp = client.models.generate(model="gemini-2.5-flash", prompt=prompt, max_output_tokens=400)
        text = getattr(resp, "text", str(resp))
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            return json.dumps({"answer": text, "contexts":[c["chunk"] for c in contexts], "reasoning":"Model output returned raw text."}, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Failed to call genai: {e}") from e
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--mock", action="store_true", help="Use mock mode (no API calls)")
    args = parser.parse_args()
    df = load_data(args.csv)
    chunks, meta = build_chunks(df)
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embed_chunks(model, chunks)
    index = build_faiss_index(embeddings)
    hits = retrieve(args.query, model, index, meta, top_k=FAISS_TOP_K)
    print("Retrieved contexts:")
    for h in hits:
        print("-", h['title'])
    out = generate_answer(args.query, hits, mock=args.mock)
    print("\\nJSON Answer:\\n", out)
if __name__ == '__main__':
    main()
