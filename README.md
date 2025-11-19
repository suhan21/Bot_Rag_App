# Mini-RAG Telegram Bot

A minimal Mini-RAG (Retrieval-Augmented Generation) Telegram bot that:
- Indexes local `docs/` (.md/.txt) into embeddings
- Answers `/ask <question>` using top-k retrieved chunks + a small LLM (OpenAI by default)

## Files
- `bot_mini_rag.py` - single-file bot that includes indexing, retrieval, and Telegram handlers
- `requirements.txt` - python dependencies
- `docs/` - sample docs folder (put your .md/.txt files here)
- `Dockerfile` - optional Dockerfile

## Quick start (Linux / macOS)
1. Create and activate a venv:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare docs:
   - Create `docs/` and add 3–5 `.md` or `.txt` files. Sample files are provided.
4. Set environment variables:
   ```bash
   export TELEGRAM_TOKEN="<your-telegram-bot-token>"
   export OPENAI_API_KEY="<your-openai-api-key>"  # optional; required for better answers
   ```
5. Run the bot:
   ```bash
   python bot_mini_rag.py
   ```
6. In Telegram, send `/ask <your question>` to the bot.
