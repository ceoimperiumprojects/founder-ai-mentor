# Founder KB

**AI-powered knowledge base CLI for founders** — semantic search over startup literature, powered by local embeddings.

Stop googling startup advice at 2 AM. Build a personal knowledge base from books, articles, and notes — then search it with meaning, not just keywords.

---

## What You Get

A **complete knowledge management CLI** with 24 commands covering search, ingestion, organization, and analytics — all running locally, no API keys needed.

### The knowledge base covers:

- **Offer Creation & Pricing** — value stacking, pricing psychology, guarantee structures
- **Lead Generation & Growth** — cold/warm outreach, content strategies, paid ads
- **Idea Validation & Lean Methodology** — MVP design, build-measure-learn, pivot signals
- **Customer Development** — discovery, validation, creation, building
- **Product Positioning** — competitive positioning, category design
- **Negotiation & Sales** — tactical empathy, calibrated questions, FBI-level techniques
- **Traction & Distribution** — 19 traction channels mapped out
- **Habit-Forming Products** — trigger-action-reward-investment loops
- **Contrarian Thinking & Monopoly Strategy** — zero-to-one mindset
- **Founder Mindset & Resilience** — handling rejection, staying focused

---

## Quick Setup

```bash
git clone https://github.com/ceoimperiumprojects/founder-ai-mentor.git
cd founder-ai-mentor
pip install -e .
```

The knowledge base comes pre-loaded. Start searching immediately:

```bash
founder-kb search "how to validate a startup idea"
founder-kb sources
founder-kb stats
```

---

## Commands

### Search & Discovery

```bash
founder-kb search "query"                          # semantic search
founder-kb keyword "term"                          # exact substring match
founder-kb multi-search "q1" "q2" "q3"             # multi-query, deduplicated
founder-kb search-tag "query" --tag books           # filter by source tags
founder-kb compare "topic" --source1 X --source2 Y  # side-by-side comparison
founder-kb duplicates "text"                        # find similar content
founder-kb random                                   # random chunk for discovery
```

### Ingestion

```bash
founder-kb add "text content" --source "my-notes"   # add raw text
founder-kb ingest path/to/book.pdf                  # ingest PDF/EPUB/TXT
founder-kb ingest-url "https://..."                 # ingest a web page
founder-kb ingest-dir path/to/books/                # batch ingest directory
founder-kb replace "source" new-file.pdf            # atomic replace
```

### Browse & Export

```bash
founder-kb chunks "source" --page 2                 # paginated chunks
founder-kb neighbors "source" 5                     # chunk with context
founder-kb export "source" --output out.txt         # export full text
founder-kb overview                                 # knowledge map
```

### Organization

```bash
founder-kb tag "source" --add fundraising --add pitch  # add tags
founder-kb tag "source" --remove old-tag               # remove tags
founder-kb annotate "source" 3 "important insight"     # annotate chunk
founder-kb sources                                     # list all sources
founder-kb delete "source" --confirm                   # delete source
```

### Analytics

```bash
founder-kb stats                                    # DB statistics
founder-kb quality                                  # chunk quality report
founder-kb reindex "source" --no-semantic            # re-chunk source
founder-kb clusters --n-clusters 8                   # semantic clustering
```

### Global Options

```bash
founder-kb --json search "query"      # JSON output (for scripting)
founder-kb --db-path ./my-db search   # custom DB path
```

---

## Add Your Own Knowledge

```bash
# Books
founder-kb ingest ~/books/lean-startup.pdf
founder-kb ingest ~/books/zero-to-one.epub

# Articles
founder-kb ingest-url "https://paulgraham.com/startupideas.html"

# Notes
founder-kb add "My key takeaway: always talk to customers first" --source "my-notes"

# Bulk import
founder-kb ingest-dir ~/startup-library/

# Organize
founder-kb tag "lean-startup" --add validation --add methodology
founder-kb tag "zero-to-one" --add strategy --add contrarian
```

---

## Tech Stack

- **Typer + Rich** — CLI framework with beautiful output
- **sentence-transformers** — local embeddings (all-MiniLM-L6-v2), no API keys
- **ChromaDB** — persistent vector database
- **LangChain** — semantic text chunking
- **scikit-learn** — topic clustering

---

## License

MIT — use it, fork it, build on it.

---

## Author

**Pavle Andjelkovic** — [Imperium Tech](https://github.com/ceoimperiumprojects) · [LinkedIn](https://www.linkedin.com/in/pavle-an%C4%91elkovi%C4%87-1614b1373)
