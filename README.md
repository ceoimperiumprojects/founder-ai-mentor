# Founder AI Mentor

An AI-powered startup mentor that gives you grounded, actionable advice using semantic search and the Model Context Protocol (MCP).

Ask Claude anything about building a startup — and get answers backed by real frameworks and proven strategies.

## How It Works

```
You ask a question → Claude searches the knowledge base → Returns grounded advice with sources
```

The system uses **sentence-transformers** to create semantic embeddings, stores them in **ChromaDB**, and exposes the search as an **MCP tool** that Claude Desktop can call directly.

The knowledge base comes **pre-built** — clone and start asking questions immediately.

## Topics Covered

Offers & pricing, lead generation, going from zero to one, lean methodology, customer development, traction channels, product positioning, habit-forming products, negotiation tactics, and customer interviews.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/ceoimperiumprojects/founder-ai-mentor.git
cd founder-ai-mentor

python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Connect to Claude Desktop

Add this to your Claude Desktop MCP config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "startup-knowledge": {
      "command": "python",
      "args": ["mcp_server/mcp_stdio_server.py"],
      "cwd": "/path/to/founder-ai-mentor"
    }
  }
}
```

Restart Claude Desktop. You can now ask questions like:

- *"How do I validate my startup idea?"*
- *"What makes an irresistible offer?"*
- *"How should I position my product in a crowded market?"*

### Optional: Add your own content

You can expand the knowledge base by adding your own PDFs or EPUBs:

```bash
# Add files to data/raw/, then:
python scripts/ingest_pdf.py --dir data/raw/
python scripts/ingest_epub.py --dir data/raw/
```

## Project Structure

```
founder-ai-mentor/
├── src/                  # Core modules
│   ├── pdf_processor.py  # PDF text extraction
│   ├── epub_processor.py # EPUB text extraction
│   ├── chunker.py        # Semantic text chunking
│   ├── embeddings.py     # Sentence-transformer embeddings
│   ├── vector_store.py   # ChromaDB vector store
│   └── search.py         # Search logic
├── scripts/              # Ingestion scripts
│   ├── ingest_pdf.py     # PDF ingestion CLI
│   └── ingest_epub.py    # EPUB ingestion CLI
├── mcp_server/           # MCP server implementations
│   ├── mcp_stdio_server.py  # stdio server for Claude Desktop
│   └── server.py            # FastAPI HTTP server
├── gui/                  # Streamlit GUI
│   └── app.py
├── docs/                 # Landing page (GitHub Pages)
└── chroma_db/            # Pre-built vector knowledge base
```

## MCP Tools

The server exposes two tools to Claude:

| Tool | Description |
|------|-------------|
| `search_startup_knowledge` | Semantic search across the knowledge base |
| `get_knowledge_stats` | Returns knowledge base statistics |

## Alternative: HTTP API

You can also run the FastAPI server for direct HTTP access:

```bash
python mcp_server/server.py
```

**Endpoints:**

- `POST /search` — Semantic search (`{"query": "...", "k": 5}`)
- `GET /health` — Health check and document count
- `GET /` — Server info

## Tech Stack

- **Python 3.10+**
- **sentence-transformers** (all-MiniLM-L6-v2) — Embeddings
- **ChromaDB** — Vector database
- **LangChain** — Semantic text chunking
- **FastAPI** — HTTP server
- **Streamlit** — GUI
- **pypdf** / **ebooklib** — Document processing

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**Pavle Andjelkovic** — [Imperium Tech](https://github.com/ceoimperiumprojects) · [LinkedIn](https://www.linkedin.com/in/pavle-an%C4%91elkovi%C4%87-1614b1373)
