# 🚀 Founder AI Mentor

**Your personal AI startup advisor — built on 10+ books worth of battle-tested knowledge, 24 intelligent tools, and Claude.**

Stop googling startup advice at 2 AM. This MCP server gives Claude direct access to a massive curated knowledge base covering everything you need to go from idea to traction — and it doesn't just search, it *understands*.

Clone. Connect. Build something great.

---

## 🔥 What You Get

This isn't a chatbot with generic advice. This is a **complete startup brain** that Claude taps into to give you real, proven answers.

### Here's what the knowledge base covers:

🎯 **Offer Creation & Pricing** — How to craft offers so compelling that people feel stupid saying no. Value stacking, pricing psychology, guarantee structures, and the exact frameworks that turn "maybe" into "shut up and take my money."

📈 **Lead Generation & Growth** — Proven systems for getting leads at scale. Cold outreach, warm outreach, content strategies, paid ads, affiliate models, and how to build lead magnets that actually convert.

🧪 **Idea Validation & Lean Methodology** — How to test your idea before burning through your savings. MVP design, build-measure-learn loops, pivot signals, and how to talk to customers without leading them to lie to you.

👥 **Customer Development** — The science of finding your first customers and understanding what they actually need. Customer discovery, validation, creation, and building — step by step.

🏆 **Product Positioning & Differentiation** — How to make your product the obvious choice in a crowded market. Competitive positioning, category design, and the frameworks that turn "me too" products into category leaders.

🤝 **Negotiation & Sales** — Tactical empathy, mirroring, labeling, calibrated questions — the FBI-level techniques that work just as well in fundraising meetings as hostage situations.

🧲 **Traction & Distribution** — 19 traction channels mapped out with frameworks for testing each one. Know exactly where to focus when you need users NOW.

🧠 **Habit-Forming Products** — The psychology of building products people can't stop using. Trigger-action-reward-investment loops that create genuine engagement, not manipulation.

💡 **Contrarian Thinking & Monopoly Strategy** — Why competition is for losers, how to find secrets hiding in plain sight, and the zero-to-one mindset that separates transformative startups from incremental ones.

🔥 **Founder Mindset & Resilience** — The mental game nobody talks about. How to handle rejection, push through the dip, stay focused when everything is on fire, and build unshakeable confidence as a founder.

---

## ⚡ 24 Tools — Not Just Search

This isn't a basic RAG. It's a **full knowledge management system** with 24 MCP tools:

### 🔍 Search & Discovery
| Tool | What it does |
|------|-------------|
| `search_startup_knowledge` | Semantic search — finds answers even when you don't use the exact words |
| `keyword_search` | Exact text matching — find specific terms, names, or phrases |
| `multi_search` | Multiple queries at once, deduplicated — catches what single search misses |
| `search_by_tag` | Filter search by category tags (e.g. only "fundraising" sources) |
| `compare_sources` | Side-by-side comparison — what do two different sources say about the same topic? |
| `find_duplicates` | Check if content already exists before adding |
| `random_chunk` | Serendipity engine — surface random insights you'd never search for |

### 📥 Ingestion
| Tool | What it does |
|------|-------------|
| `add_knowledge` | Ingest raw text directly |
| `ingest_file` | Ingest PDF, EPUB, or TXT with automatic extraction and chunking |
| `ingest_url` | Fetch and ingest any web page — blogs, articles, docs |
| `batch_ingest_directory` | Point at a folder, ingest everything — bulk loading |
| `replace_source` | Atomically swap a source's content — delete old, add new |

### 📖 Browse & Export
| Tool | What it does |
|------|-------------|
| `get_source_chunks` | Browse stored chunks with pagination |
| `get_neighboring_chunks` | Expand search results with surrounding context |
| `export_source` | Reconstruct and export a full source as text |
| `knowledge_overview` | High-level map of the entire knowledge base |

### 🏷️ Organization
| Tool | What it does |
|------|-------------|
| `tag_source` | Add tags to sources — organize at scale |
| `annotate_chunk` | Add personal notes to specific passages |
| `list_sources` | See all sources with chunk counts |
| `delete_by_source` | Clean removal of any source |

### 📊 Analytics & Intelligence
| Tool | What it does |
|------|-------------|
| `get_knowledge_stats` | Database statistics at a glance |
| `chunk_quality_report` | Health check — finds broken or garbled chunks |
| `reindex_source` | Re-chunk with different parameters without re-uploading |
| `semantic_clusters` | AI topic clustering — see what's covered and where the gaps are |

---

## ⚡ Quick Setup

### 1. Clone & install

```bash
git clone https://github.com/ceoimperiumprojects/founder-ai-mentor.git
cd founder-ai-mentor
pip install -r requirements.txt
```

### 2. Connect to Claude Desktop

Add to your `claude_desktop_config.json`:

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

### 3. Restart Claude Desktop

That's it. The knowledge base comes pre-loaded. Start asking questions immediately.

---

## 💬 Ask It Anything

- *"How do I validate my startup idea before writing any code?"*
- *"What makes an irresistible offer?"*
- *"How should I position my product in a crowded market?"*
- *"What are the best traction channels for early-stage startups?"*
- *"What negotiation tactics can I use in my next fundraising meeting?"*
- *"How do I build a habit-forming product?"*
- *"Give me a random insight — surprise me."*

---

## 🧩 Make It Yours

Add your own books, articles, notes, or anything:

```
"Ingest this PDF: C:/path/to/my-book.pdf"
"Ingest this URL: https://example.com/great-article"
"Add this to the knowledge base as 'my-notes': [paste text]"
"Ingest all files from C:/my-startup-library/"
```

Organize and analyze:

```
"Tag 'my-notes' with: fundraising, pitch"
"Run a quality report on the whole knowledge base"
"Show me semantic clusters — what topics do we cover?"
```

---

## 🛠️ Tech Stack

- **Python** — core runtime
- **sentence-transformers** — local embeddings, no API keys needed
- **ChromaDB** — persistent vector database
- **LangChain** — semantic text chunking
- **scikit-learn** — topic clustering
- **MCP** — Claude Desktop integration

---

## 📄 License

MIT — use it, fork it, build on it.

---

## 👤 Author

**Pavle Andjelkovic** — [Imperium Tech](https://github.com/ceoimperiumprojects) · [LinkedIn](https://www.linkedin.com/in/pavle-an%C4%91elkovi%C4%87-1614b1373)
