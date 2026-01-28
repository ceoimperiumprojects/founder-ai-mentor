# Startup Knowledge RAG

AI sistem za semanticko pretrazivanje startup literature.

## Instalacija

```bash
# Kreiraj virtual environment
python -m venv venv

# Aktiviraj (Windows)
.\venv\Scripts\activate

# Aktiviraj (Linux/Mac)
source venv/bin/activate

# Instaliraj dependencies
pip install -r requirements.txt
```

## Dodavanje znanja

### CLI
```bash
# Jedan PDF
python scripts/ingest_pdf.py --pdf data/raw/startup_owners_manual.pdf

# Svi PDF-ovi iz foldera
python scripts/ingest_pdf.py --dir data/raw/
```

### GUI
```bash
streamlit run gui/app.py
```

## MCP Server

```bash
# Pokreni server
uvicorn mcp_server.server:app --reload --port 8000

# Ili direktno
python mcp_server/server.py
```

### API Endpoints

- `GET /` - Info
- `GET /health` - Health check
- `POST /search` - Semanticka pretraga

### Primjer pretrage

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How to validate startup idea?", "k": 5}'
```

## Claude Integracija

Dodaj MCP server u Claude Desktop config:

```json
{
  "mcpServers": {
    "startup-knowledge": {
      "command": "uvicorn",
      "args": ["mcp_server.server:app", "--port", "8000"],
      "cwd": "C:/Users/Korisnik/Desktop/MCP Founder OS/startup-knowledge-rag"
    }
  }
}
```

## Tech Stack

- Python 3.10+
- pypdf - PDF processing
- LangChain - Text splitting
- sentence-transformers - Embeddings (all-MiniLM-L6-v2)
- ChromaDB - Vector database
- FastAPI - MCP Server
- Streamlit - GUI

## Autor

Pavle Andjelkovic - Imperium Tech
