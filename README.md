# Founder AI Mentor

AI startup mentor for Claude Desktop. Pre-built knowledge base covering offers, pricing, lead generation, lean methodology, customer development, positioning, negotiation, and more.

Clone, connect, ask.

## Setup

```bash
git clone https://github.com/ceoimperiumprojects/founder-ai-mentor.git
cd founder-ai-mentor
pip install -r requirements.txt
```

Add to your Claude Desktop config (`claude_desktop_config.json`):

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

Restart Claude Desktop. Done.

## Example Questions

- *"How do I validate my startup idea?"*
- *"What makes an irresistible offer?"*
- *"How should I position my product in a crowded market?"*
- *"What are the best traction channels for early-stage startups?"*

## Expand the Knowledge Base

Add your own PDFs or EPUBs:

```bash
python scripts/ingest_pdf.py --dir data/raw/
python scripts/ingest_epub.py --dir data/raw/
```

## Tech Stack

Python, sentence-transformers, ChromaDB, LangChain, MCP

## License

MIT

## Author

**Pavle Andjelkovic** — [Imperium Tech](https://github.com/ceoimperiumprojects) · [LinkedIn](https://www.linkedin.com/in/pavle-an%C4%91elkovi%C4%87-1614b1373)
