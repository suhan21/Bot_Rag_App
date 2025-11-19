#!/usr/bin/env python3
"""
bot_mini_rag.py
Single-file Mini-RAG Telegram bot.

Features:
- Indexes local ./docs/*.md and ./docs/*.txt into an sqlite DB with sentence-transformers embeddings.
- /ask <query> : retrieves top-k chunks and uses OpenAI (if configured) to generate an answer.
- Robust fallback: if OpenAI isn't available, returns an extractive, readable summary.

Usage:
1. Put docs in ./docs (3-5 files for assignment).
2. Set TELEGRAM_TOKEN and optionally OPENAI_API_KEY (env vars or .env).
3. Run: python bot_mini_rag.py
"""
import os
import sqlite3
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional
import re
import sys
import json

# Optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Telegram
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Embeddings
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Config
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

DOCS_DIR = Path("docs")
DB_PATH = "embeddings.db"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Initialize embedding model (may download on first run)
print("Loading embedding model... (this may take a moment)")
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# --- sqlite embedding store ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        doc_name TEXT,
        chunk_index INTEGER,
        text TEXT,
        embedding BLOB
    )""")
    conn.commit()
    conn.close()

def upsert_chunk(id: str, doc_name: str, index: int, text: str, emb_bytes: bytes):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("REPLACE INTO chunks (id, doc_name, chunk_index, text, embedding) VALUES (?, ?, ?, ?, ?)",
                (id, doc_name, index, text, emb_bytes))
    conn.commit()
    conn.close()

def get_all_embeddings() -> List[Tuple[str, str, int, str, bytes]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, doc_name, chunk_index, text, embedding FROM chunks")
    rows = cur.fetchall()
    conn.close()
    return rows

# --- robust chunker ---
def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text:
        return []
    chunks = []
    pos = 0
    text_len = len(text)
    while pos < text_len:
        end = min(pos + size, text_len)
        chunk = text[pos:end]
        # avoid mid-word split if possible
        if end < text_len and not text[end].isspace():
            back = chunk.rfind(" ")
            if back != -1 and back >= max(0, size - 200):
                chunk = chunk[:back]
                end = pos + back
        chunk_clean = ""
        try:
            chunk_clean = chunk.strip()
        except Exception:
            chunk_clean = str(chunk).strip()
        if chunk_clean:
            chunks.append(chunk_clean)
        next_pos = end - overlap
        if next_pos <= pos:
            next_pos = pos + max(1, size - overlap)
        pos = next_pos
    return chunks

# --- index local docs ---
def index_docs(force: bool = False):
    init_db()
    docs = list(DOCS_DIR.glob("*.md")) + list(DOCS_DIR.glob("*.txt"))
    if not docs:
        print("No docs found in ./docs — create some .md or .txt files.")
        return
    for doc in docs:
        try:
            text = doc.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"[index_docs] Failed to read {doc.name}: {e}")
            continue
        if not text or not text.strip():
            print(f"[index_docs] Skipping empty file: {doc.name}")
            continue
        try:
            chunks = chunk_text(text)
        except Exception as e:
            print(f"[index_docs] Chunking failed for {doc.name}: {e}")
            continue
        if not chunks:
            print(f"[index_docs] No chunks for {doc.name} (skipping).")
            continue
        for i, c in enumerate(chunks):
            try:
                uid = hashlib.sha256((str(doc.name) + str(i) + (c[:64] if c else "")).encode()).hexdigest()
                emb = embedder.encode(c, convert_to_numpy=True)
                emb_bytes = emb.tobytes()
                upsert_chunk(uid, doc.name, i, c, emb_bytes)
            except Exception as e:
                print(f"[index_docs] Failed to index chunk {i} of {doc.name}: {e}")
                continue
    print("Indexing complete.")

# --- retrieval ---
def retrieve(query: str, top_k: int = TOP_K):
    q_emb = embedder.encode(query, convert_to_numpy=True)
    rows = get_all_embeddings()
    if not rows:
        return []
    ids, doc_names, idxs, texts, emb_blobs = zip(*rows)
    embs = np.array([np.frombuffer(b, dtype=np.float32) for b in emb_blobs])
    scores = util.cos_sim(q_emb, embs)[0].cpu().numpy()
    top_indices = np.argsort(-scores)[:top_k]
    results = []
    for ti in top_indices:
        results.append({
            "doc_name": doc_names[ti],
            "chunk_index": idxs[ti],
            "text": texts[ti],
            "score": float(scores[ti])
        })
    return results

# --- helper to extract content from various OpenAI response shapes ---
def extract_openai_text(resp) -> Optional[str]:
    """
    Try multiple ways to extract the assistant text from response objects/dicts
    returned by different OpenAI SDK versions.
    """
    try:
        # dict-style
        if isinstance(resp, dict):
            return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    try:
        # object-style: resp.choices[0].message.content or ['content']
        c = resp.choices[0]
        # try a few attribute accesses
        if hasattr(c, "message"):
            m = c.message
            # m might be a dict or object
            if isinstance(m, dict) and "content" in m:
                return m["content"].strip()
            if hasattr(m, "get") and m.get("content"):
                return m.get("content").strip()
            if hasattr(m, "content"):
                return m.content.strip()
        # older old sdk maybe: c['message']['content']
        if isinstance(c, dict):
            return c["message"]["content"].strip()
    except Exception:
        pass
    # last resort: try text attribute
    try:
        if hasattr(resp, "text"):
            return resp.text.strip()
    except Exception:
        pass
    return None

# --- LLM call supporting both new and old OpenAI SDKs ---
def call_llm(prompt: str, snippets: List[str] = None) -> str:
    """
    Try:
      1) NEW OpenAI SDK: from openai import OpenAI -> client.chat.completions.create(...)
      2) OLD OpenAI SDK: import openai -> openai.ChatCompletion.create(...)
    On any failure, fallback to a short extractive summary built from snippets.
    """
    if OPENAI_API_KEY:
        # Try new SDK first
        try:
            try:
                from openai import OpenAI  # new SDK
                client = OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=400,
                    temperature=0.2,
                )
                text = extract_openai_text(response)
                if text:
                    return text
            except Exception as e_new:
                # Try old SDK
                try:
                    import openai
                    openai.api_key = OPENAI_API_KEY
                    resp = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=400,
                        temperature=0.2,
                    )
                    text = extract_openai_text(resp)
                    if text:
                        return text
                except Exception as e_old:
                    print("[call_llm] Both new and old OpenAI SDK calls failed. NewSDKError:", e_new, "OldSDKError:", e_old)
        except Exception as e_outer:
            print("[call_llm] Unexpected error while calling OpenAI:", e_outer)

    # Fallback: extractive summary (safe, requires no external API)
    if snippets:
        # naive sentence extraction and scoring by length
        all_text = "\n\n".join(snippets)
        sentences = re.split(r'(?<=[.!?])\s+', all_text)
        seen = set()
        scored = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            key = s_clean.lower()
            if key in seen:
                continue
            seen.add(key)
            scored.append((len(s_clean), s_clean))
        scored.sort(reverse=True)
        top = [s for _, s in scored[:4]]
        if top:
            return "⚠️ (No LLM available) Extractive summary from retrieved snippets:\n\n" + " ".join(top)
    return "⚠️ LLM unavailable and no snippets to summarize."

# --- Telegram handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Mini-RAG bot ready.\nUse /ask <your question>\nPlace your docs in ./docs as .md or .txt and run index_docs() before starting."
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("/ask <query>  - ask a question\n/help - show this help")

async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /ask <your question>")
        return
    query = " ".join(context.args)
    await update.message.reply_text("Searching docs... (this may take a second)")
    results = retrieve(query, top_k=TOP_K)
    if not results:
        await update.message.reply_text("No docs indexed. Ask the admin to run indexing (index_docs()).")
        return
    prompt_lines = [
        "You are a helpful assistant. Use the provided context to answer the question concisely.",
        f"Question: {query}",
        "\nContext snippets (most relevant first):"
    ]
    snippets = []
    for r in results:
        snippet = r["text"].strip()
        prompt_lines.append(f"--- Source: {r['doc_name']} (score={r['score']:.3f}) ---\n{snippet}\n")
        snippets.append(snippet)
    prompt_lines.append("\nInstructions: Answer in 2-6 sentences. If answer is not contained in context say so.")
    prompt = "\n\n".join(prompt_lines)
    answer = call_llm(prompt, snippets=snippets)
    sources = ", ".join(sorted({r["doc_name"] for r in results}))
    await update.message.reply_text(f"{answer}\n\nSources: {sources}")

# --- Entry point ---
def main():
    # Index docs at startup
    index_docs(force=False)
    token = TELEGRAM_TOKEN
    if not token:
        print("Set TELEGRAM_TOKEN env var and re-run.")
        return
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    print("Bot started. Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == "__main__":
    main()
