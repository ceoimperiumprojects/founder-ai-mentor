# Founder AI Mentor

An AI-powered startup mentor that gives you grounded, actionable advice by searching through 10 essential startup books using semantic search and the Model Context Protocol (MCP).

Ask Claude anything about building a startup — and get answers backed by real frameworks from Hormozi, Thiel, Ries, and more.

## How It Works

```
You ask a question → Claude searches the knowledge base → Returns grounded advice with sources
```

The system uses **sentence-transformers** to create semantic embeddings of book content, stores them in **ChromaDB**, and exposes the search as an **MCP tool** that Claude Desktop can call directly.

## Books Included

| Book | Author |
|------|--------|
| $100M Offers | Alex Hormozi |
| $100M Leads | Alex Hormozi |
| Zero to One | Peter Thiel |
| The Lean Startup | Eric Ries |
| The Startup Owner's Manual | Steve Blank |
| Traction | Gabriel Weinberg & Justin Mares |
| Obviously Awesome | April Dunford |
| Hooked | Nir Eyal |
| Never Split the Difference | Chris Voss |
| The Mom Test | Rob Fitzpatrick |

> You supply your own book files (PDF or EPUB). This repo includes the ingestion pipeline and search infrastructure only.

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

### 2. Add your books

Place PDF or EPUB files into `data/raw/`, then ingest:

```bash
# Ingest all PDFs
python scripts/ingest_pdf.py --dir data/raw/

# Ingest all EPUBs
python scripts/ingest_epub.py --dir data/raw/
```

### 3. Connect to Claude Desktop

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
- *"What makes an irresistible offer according to Hormozi?"*
- *"How should I position my product in a crowded market?"*

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
├── data/raw/             # Your book files go here (gitignored)
└── chroma_db/            # Vector database (gitignored)
```

## MCP Tools

The server exposes two tools to Claude:

| Tool | Description |
|------|-------------|
| `search_startup_knowledge` | Semantic search across all ingested books |
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

**Pavle Andjelkovic** — [Imperium Tech](https://github.com/ceoimperiumprojects)
