#!/usr/bin/env python3
"""
MCP Server za Startup Knowledge RAG - stdio verzija za Claude Desktop.
"""

import sys
import json
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Lazy-load vector store to avoid blocking the MCP handshake
project_root = Path(__file__).parent.parent
db_path = project_root / "chroma_db"
vector_store = None

# Sidecar JSON file for tags and annotations
META_FILE = project_root / "knowledge_meta.json"


def _load_meta() -> dict:
    """Load tags/annotations metadata from JSON file."""
    if META_FILE.exists():
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    return {"tags": {}, "annotations": {}}


def _save_meta(data: dict):
    """Save tags/annotations metadata to JSON file."""
    META_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_vector_store():
    global vector_store
    if vector_store is None:
        sys.stderr.write("Loading embedding model...\n")
        sys.stderr.flush()
        from src.vector_store import VectorStore
        vector_store = VectorStore(persist_directory=str(db_path))
        sys.stderr.write("Model loaded and ready!\n")
        sys.stderr.flush()
    return vector_store


def handle_request(request):
    """Handle incoming MCP request."""
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "startup-knowledge-rag",
                    "version": "1.0.0"
                }
            }
        }

    elif method == "notifications/initialized":
        return None  # No response needed for notifications

    elif method == "notifications/cancelled":
        return None  # No response needed for notifications

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "search_startup_knowledge",
                        "description": "Search through startup literature and books for relevant information. Use this to find advice about startups, customer development, validation, business models, etc.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query - what you want to learn about startups"
                                },
                                "num_results": {
                                    "type": "integer",
                                    "description": "Number of results to return (default: 5)",
                                    "default": 5
                                },
                                "source_filter": {
                                    "type": "string",
                                    "description": "Optional: only search within a specific source (use list_sources to see available sources)"
                                }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "get_knowledge_stats",
                        "description": "Get statistics about the startup knowledge base",
                        "inputSchema": {
                            "type": "object",
                            "properties": {}
                        }
                    },
                    {
                        "name": "add_knowledge",
                        "description": "Ingest text into the startup knowledge base. The text will be automatically chunked and embedded for later retrieval via search_startup_knowledge.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "The text content to ingest into the knowledge base"
                                },
                                "source": {
                                    "type": "string",
                                    "description": "A name/label for this source (e.g. 'founders-guide-unshakeable-mind')"
                                },
                                "use_semantic_chunking": {
                                    "type": "boolean",
                                    "description": "Use semantic chunking (true) or character-based chunking (false). Default: true",
                                    "default": True
                                }
                            },
                            "required": ["content", "source"]
                        }
                    },
                    {
                        "name": "list_sources",
                        "description": "List all sources in the knowledge base with their chunk counts. Use this to see what has been ingested.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {}
                        }
                    },
                    {
                        "name": "delete_by_source",
                        "description": "Delete all chunks belonging to a specific source from the knowledge base.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string",
                                    "description": "The source name to delete (use list_sources to see available sources)"
                                }
                            },
                            "required": ["source"]
                        }
                    },
                    {
                        "name": "ingest_file",
                        "description": "Ingest a file (PDF, EPUB, or TXT) from disk into the knowledge base. Extracts text, chunks it, and embeds it automatically.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Absolute path to the file to ingest"
                                },
                                "source": {
                                    "type": "string",
                                    "description": "Optional source label. If omitted, the filename (without extension) is used."
                                },
                                "use_semantic_chunking": {
                                    "type": "boolean",
                                    "description": "Use semantic chunking (true) or character-based chunking (false). Default: true",
                                    "default": True
                                }
                            },
                            "required": ["file_path"]
                        }
                    },
                    {
                        "name": "ingest_url",
                        "description": "Fetch a web page, extract its text content, and ingest it into the knowledge base. Works with articles, blog posts, documentation pages, etc.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "description": "The URL to fetch and ingest"
                                },
                                "source": {
                                    "type": "string",
                                    "description": "Optional source label. If omitted, the domain + path is used."
                                },
                                "use_semantic_chunking": {
                                    "type": "boolean",
                                    "description": "Use semantic chunking (true) or character-based chunking (false). Default: true",
                                    "default": True
                                }
                            },
                            "required": ["url"]
                        }
                    },
                    {
                        "name": "get_source_chunks",
                        "description": "Browse the actual stored chunks for a specific source. Use this to read/verify what was ingested, or to review content without searching.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string",
                                    "description": "The source name to browse (use list_sources to see available sources)"
                                },
                                "offset": {
                                    "type": "integer",
                                    "description": "Skip this many chunks (for pagination). Default: 0",
                                    "default": 0
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Max chunks to return. Default: 10",
                                    "default": 10
                                }
                            },
                            "required": ["source"]
                        }
                    },
                    {
                        "name": "find_duplicates",
                        "description": "Check if similar content already exists in the knowledge base. Use before ingesting to avoid redundancy.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "The text to check for duplicates"
                                },
                                "threshold": {
                                    "type": "number",
                                    "description": "Similarity threshold (0-1, where 1 = identical). Default: 0.85",
                                    "default": 0.85
                                },
                                "num_results": {
                                    "type": "integer",
                                    "description": "Max number of similar chunks to return. Default: 5",
                                    "default": 5
                                }
                            },
                            "required": ["text"]
                        }
                    },
                    {
                        "name": "knowledge_overview",
                        "description": "Get a high-level overview of the entire knowledge base: all sources with chunk counts and sample previews from each. Like a table of contents for the knowledge base.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {}
                        }
                    },
                    {
                        "name": "get_neighboring_chunks",
                        "description": "Get a specific chunk plus its surrounding context (neighboring chunks before and after). Use this to expand search results with more context.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string",
                                    "description": "The source name (use list_sources to see available sources)"
                                },
                                "chunk_index": {
                                    "type": "integer",
                                    "description": "The index of the chunk to retrieve"
                                },
                                "context": {
                                    "type": "integer",
                                    "description": "Number of neighboring chunks before and after. Default: 2",
                                    "default": 2
                                }
                            },
                            "required": ["source", "chunk_index"]
                        }
                    },
                    {
                        "name": "multi_search",
                        "description": "Execute multiple search queries at once and get deduplicated results. More comprehensive than a single query. If queries param is empty/null, auto-generates variations of the main query.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The main search query"
                                },
                                "queries": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional: additional related queries. If omitted, auto-generates variations."
                                },
                                "num_results": {
                                    "type": "integer",
                                    "description": "Max results to return after deduplication. Default: 10",
                                    "default": 10
                                },
                                "source_filter": {
                                    "type": "string",
                                    "description": "Optional: only search within a specific source"
                                }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "replace_source",
                        "description": "Atomically replace all content for a source. Deletes existing chunks and ingests new content in one operation. Supports PDF, EPUB, TXT files, URLs, or raw text.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string",
                                    "description": "The source to replace"
                                },
                                "content_type": {
                                    "type": "string",
                                    "enum": ["file", "url", "text"],
                                    "description": "Type of content being provided"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "File path, URL, or raw text content depending on content_type"
                                },
                                "use_semantic_chunking": {
                                    "type": "boolean",
                                    "description": "Use semantic chunking. Default: true",
                                    "default": True
                                }
                            },
                            "required": ["source", "content_type", "content"]
                        }
                    },
                    {
                        "name": "batch_ingest_directory",
                        "description": "Ingest all supported files (.pdf, .epub, .txt) from a directory. Each file becomes a separate source. Returns summary of all processed files.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "directory": {
                                    "type": "string",
                                    "description": "Absolute path to the directory to scan"
                                },
                                "use_semantic_chunking": {
                                    "type": "boolean",
                                    "description": "Use semantic chunking for all files. Default: true",
                                    "default": True
                                }
                            },
                            "required": ["directory"]
                        }
                    },
                    {
                        "name": "keyword_search",
                        "description": "Search chunks by exact keyword/substring match (not semantic). Useful for finding specific terms, names, or phrases.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "keyword": {
                                    "type": "string",
                                    "description": "The keyword or phrase to search for (exact substring match)"
                                },
                                "num_results": {
                                    "type": "integer",
                                    "description": "Max results to return. Default: 10",
                                    "default": 10
                                },
                                "source_filter": {
                                    "type": "string",
                                    "description": "Optional: only search within a specific source"
                                }
                            },
                            "required": ["keyword"]
                        }
                    },
                    {
                        "name": "export_source",
                        "description": "Export the full reconstructed text of a source by joining all its chunks in order. Useful for review or re-ingestion.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string",
                                    "description": "The source name to export"
                                }
                            },
                            "required": ["source"]
                        }
                    },
                    {
                        "name": "tag_source",
                        "description": "Add, remove, or set tags on a source for categorization. Tags persist in a sidecar JSON file.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string",
                                    "description": "The source name to tag"
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Tags to add, remove, or set"
                                },
                                "action": {
                                    "type": "string",
                                    "enum": ["add", "remove", "set"],
                                    "description": "Action: 'add' (default) appends tags, 'remove' removes them, 'set' replaces all tags",
                                    "default": "add"
                                }
                            },
                            "required": ["source", "tags"]
                        }
                    },
                    {
                        "name": "search_by_tag",
                        "description": "Semantic search filtered to sources that have ALL specified tags. Combine tagging with search for precise retrieval.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The semantic search query"
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Only search sources that have ALL these tags"
                                },
                                "num_results": {
                                    "type": "integer",
                                    "description": "Max results to return. Default: 5",
                                    "default": 5
                                }
                            },
                            "required": ["query", "tags"]
                        }
                    },
                    {
                        "name": "compare_sources",
                        "description": "Compare what two sources say about a topic. Runs semantic search on each source separately and returns side-by-side results.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "topic": {
                                    "type": "string",
                                    "description": "The topic or question to compare across sources"
                                },
                                "source_a": {
                                    "type": "string",
                                    "description": "First source name"
                                },
                                "source_b": {
                                    "type": "string",
                                    "description": "Second source name"
                                },
                                "num_results": {
                                    "type": "integer",
                                    "description": "Results per source. Default: 3",
                                    "default": 3
                                }
                            },
                            "required": ["topic", "source_a", "source_b"]
                        }
                    },
                    {
                        "name": "chunk_quality_report",
                        "description": "Analyze chunk quality: average/min/max length, short chunks, long chunks, whitespace-heavy chunks. Helps identify ingestion issues.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string",
                                    "description": "Optional: analyze a specific source. Omit for whole database."
                                }
                            }
                        }
                    },
                    {
                        "name": "reindex_source",
                        "description": "Re-chunk a source with different settings. Exports text, deletes old chunks, re-chunks and re-embeds. Note: original chunk boundaries are lost.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string",
                                    "description": "The source to reindex"
                                },
                                "use_semantic_chunking": {
                                    "type": "boolean",
                                    "description": "Use semantic chunking. Default: true",
                                    "default": True
                                },
                                "chunk_size": {
                                    "type": "integer",
                                    "description": "Chunk size for character-based chunking. Default: 1000",
                                    "default": 1000
                                },
                                "chunk_overlap": {
                                    "type": "integer",
                                    "description": "Overlap between chunks. Default: 200",
                                    "default": 200
                                }
                            },
                            "required": ["source"]
                        }
                    },
                    {
                        "name": "annotate_chunk",
                        "description": "Add a note/annotation to a specific chunk, or view existing annotations. Annotations persist in a sidecar JSON file.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string",
                                    "description": "The source name"
                                },
                                "chunk_index": {
                                    "type": "integer",
                                    "description": "The chunk index to annotate"
                                },
                                "note": {
                                    "type": "string",
                                    "description": "The annotation text. Omit to view existing annotations."
                                }
                            },
                            "required": ["source", "chunk_index"]
                        }
                    },
                    {
                        "name": "random_chunk",
                        "description": "Get random chunks from the knowledge base for discovery or spot-checking quality.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "num_chunks": {
                                    "type": "integer",
                                    "description": "Number of random chunks. Default: 1",
                                    "default": 1
                                },
                                "source": {
                                    "type": "string",
                                    "description": "Optional: only pick from this source"
                                }
                            }
                        }
                    },
                    {
                        "name": "semantic_clusters",
                        "description": "Cluster all chunks by embedding similarity using KMeans. Shows topic groupings with representative chunks. Requires scikit-learn.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "num_clusters": {
                                    "type": "integer",
                                    "description": "Number of clusters. Default: 5",
                                    "default": 5
                                },
                                "source": {
                                    "type": "string",
                                    "description": "Optional: cluster only chunks from this source"
                                }
                            }
                        }
                    }
                ]
            }
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name == "search_startup_knowledge":
            query = arguments.get("query", "")
            num_results = arguments.get("num_results", 5)
            source_filter = arguments.get("source_filter")

            try:
                vs = get_vector_store()
                results = vs.search(query, k=num_results, source_filter=source_filter)

                if not results:
                    content = "No results found for your query."
                else:
                    content_parts = []
                    for i, r in enumerate(results, 1):
                        relevance = 1 - r["distance"]
                        source = r["metadata"].get("source", "unknown")
                        text = r["text"]
                        content_parts.append(
                            f"**Result {i}** (Relevance: {relevance:.0%}, Source: {source})\n{text}"
                        )
                    content = "\n\n---\n\n".join(content_parts)

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": content
                            }
                        ]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error searching: {str(e)}"
                            }
                        ],
                        "isError": True
                    }
                }

        elif tool_name == "get_knowledge_stats":
            try:
                vs = get_vector_store()
                stats = vs.get_stats()
                content = f"Knowledge Base Statistics:\n- Total chunks: {stats['total_documents']}"
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": content
                            }
                        ]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error getting stats: {str(e)}"
                            }
                        ],
                        "isError": True
                    }
                }

        elif tool_name == "add_knowledge":
            text_content = arguments.get("content", "")
            source = arguments.get("source", "unknown")
            use_semantic = arguments.get("use_semantic_chunking", True)

            try:
                from src.chunker import create_chunks_with_metadata

                chunks = create_chunks_with_metadata(
                    text=text_content,
                    source=source,
                    use_semantic=use_semantic,
                )

                texts = [c["text"] for c in chunks]
                metadatas = [c["metadata"] for c in chunks]
                ids = [f"{source}_{c['metadata']['chunk_index']}" for c in chunks]

                vs = get_vector_store()
                vs.add_documents(texts=texts, metadatas=metadatas, ids=ids)

                stats = vs.get_stats()
                content = (
                    f"Successfully added {len(chunks)} chunks from source '{source}'.\n"
                    f"Knowledge base now contains {stats['total_documents']} total chunks."
                )

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": content
                            }
                        ]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error adding knowledge: {str(e)}"
                            }
                        ],
                        "isError": True
                    }
                }

        elif tool_name == "list_sources":
            try:
                vs = get_vector_store()
                sources = vs.list_sources()

                if not sources:
                    content = "Knowledge base is empty — no sources found."
                else:
                    lines = [f"- **{s['source']}**: {s['chunks']} chunks" for s in sources]
                    total = sum(s["chunks"] for s in sources)
                    lines.append(f"\n**Total: {len(sources)} sources, {total} chunks**")
                    content = "\n".join(lines)

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": content
                            }
                        ]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error listing sources: {str(e)}"
                            }
                        ],
                        "isError": True
                    }
                }

        elif tool_name == "delete_by_source":
            source = arguments.get("source", "")

            try:
                vs = get_vector_store()
                deleted = vs.delete_by_source(source)
                stats = vs.get_stats()
                content = (
                    f"Deleted {deleted} chunks from source '{source}'.\n"
                    f"Knowledge base now contains {stats['total_documents']} total chunks."
                )

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": content
                            }
                        ]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error deleting source: {str(e)}"
                            }
                        ],
                        "isError": True
                    }
                }

        elif tool_name == "ingest_file":
            file_path = arguments.get("file_path", "")
            use_semantic = arguments.get("use_semantic_chunking", True)

            try:
                from pathlib import Path as P
                fp = P(file_path)

                if not fp.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")

                ext = fp.suffix.lower()
                source = arguments.get("source") or fp.stem

                if ext == ".pdf":
                    from src.pdf_processor import extract_text_from_pdf
                    text = extract_text_from_pdf(file_path)
                elif ext == ".epub":
                    from src.epub_processor import extract_text_from_epub
                    text = extract_text_from_epub(file_path)
                elif ext == ".txt":
                    text = fp.read_text(encoding="utf-8")
                else:
                    raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .epub, .txt")

                from src.chunker import create_chunks_with_metadata

                chunks = create_chunks_with_metadata(
                    text=text,
                    source=source,
                    use_semantic=use_semantic,
                )

                texts = [c["text"] for c in chunks]
                metadatas = [c["metadata"] for c in chunks]
                ids = [f"{source}_{c['metadata']['chunk_index']}" for c in chunks]

                vs = get_vector_store()
                vs.add_documents(texts=texts, metadatas=metadatas, ids=ids)

                stats = vs.get_stats()
                content = (
                    f"Successfully ingested '{fp.name}' as source '{source}'.\n"
                    f"Added {len(chunks)} chunks.\n"
                    f"Knowledge base now contains {stats['total_documents']} total chunks."
                )

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": content
                            }
                        ]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error ingesting file: {str(e)}"
                            }
                        ],
                        "isError": True
                    }
                }

        elif tool_name == "ingest_url":
            url = arguments.get("url", "")
            use_semantic = arguments.get("use_semantic_chunking", True)

            try:
                import requests
                from bs4 import BeautifulSoup
                from urllib.parse import urlparse

                response = requests.get(url, timeout=30, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; StartupKnowledgeRAG/1.0)"
                })
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")

                # Remove script, style, nav, footer elements
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()

                text = soup.get_text(separator="\n", strip=True)

                if not text.strip():
                    raise ValueError("No text content could be extracted from the URL.")

                parsed = urlparse(url)
                source = arguments.get("source") or f"{parsed.netloc}{parsed.path}".rstrip("/")

                from src.chunker import create_chunks_with_metadata

                chunks = create_chunks_with_metadata(
                    text=text,
                    source=source,
                    use_semantic=use_semantic,
                )

                texts = [c["text"] for c in chunks]
                metadatas = [c["metadata"] for c in chunks]
                ids = [f"{source}_{c['metadata']['chunk_index']}" for c in chunks]

                vs = get_vector_store()
                vs.add_documents(texts=texts, metadatas=metadatas, ids=ids)

                stats = vs.get_stats()
                content = (
                    f"Successfully ingested URL as source '{source}'.\n"
                    f"Extracted ~{len(text)} characters of text.\n"
                    f"Added {len(chunks)} chunks.\n"
                    f"Knowledge base now contains {stats['total_documents']} total chunks."
                )

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": content
                            }
                        ]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error ingesting URL: {str(e)}"
                            }
                        ],
                        "isError": True
                    }
                }

        elif tool_name == "get_source_chunks":
            source = arguments.get("source", "")
            offset = arguments.get("offset", 0)
            limit = arguments.get("limit", 10)

            try:
                vs = get_vector_store()
                result = vs.get_source_chunks(source, offset=offset, limit=limit)

                if not result["chunks"]:
                    content = f"No chunks found for source '{source}'."
                else:
                    parts = []
                    for c in result["chunks"]:
                        idx = c["metadata"].get("chunk_index", "?")
                        parts.append(f"**Chunk {idx}**\n{c['text']}")
                    content = "\n\n---\n\n".join(parts)
                    content += (
                        f"\n\n**Showing {len(result['chunks'])} of {result['total']} chunks "
                        f"(offset: {result['offset']})**"
                    )

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": content
                            }
                        ]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error getting chunks: {str(e)}"
                            }
                        ],
                        "isError": True
                    }
                }

        elif tool_name == "find_duplicates":
            text = arguments.get("text", "")
            threshold = arguments.get("threshold", 0.85)
            num_results = arguments.get("num_results", 5)

            try:
                vs = get_vector_store()
                # Search with more results, then filter by threshold
                results = vs.search(text, k=num_results)

                # Convert cosine distance to similarity: similarity = 1 - distance
                duplicates = []
                for r in results:
                    similarity = 1 - r["distance"]
                    if similarity >= threshold:
                        duplicates.append({
                            "similarity": similarity,
                            "source": r["metadata"].get("source", "unknown"),
                            "chunk_index": r["metadata"].get("chunk_index", "?"),
                            "preview": r["text"][:300],
                        })

                if not duplicates:
                    content = f"No duplicates found above {threshold:.0%} similarity threshold."
                else:
                    parts = []
                    for d in duplicates:
                        parts.append(
                            f"- **{d['similarity']:.0%} match** (Source: {d['source']}, "
                            f"Chunk {d['chunk_index']})\n  {d['preview']}..."
                        )
                    content = f"Found {len(duplicates)} similar chunk(s):\n\n" + "\n\n".join(parts)

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": content
                            }
                        ]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error finding duplicates: {str(e)}"
                            }
                        ],
                        "isError": True
                    }
                }

        elif tool_name == "knowledge_overview":
            try:
                vs = get_vector_store()
                overview = vs.get_overview()

                if not overview:
                    content = "Knowledge base is empty."
                else:
                    parts = []
                    total_chunks = 0
                    for src in overview:
                        total_chunks += src["chunks"]
                        section = f"### {src['source']} ({src['chunks']} chunks)"
                        for s in src["samples"]:
                            section += f"\n- Chunk {s['index']}: _{s['preview']}_..."
                        parts.append(section)

                    header = f"**Knowledge Base Overview: {len(overview)} sources, {total_chunks} total chunks**\n"
                    content = header + "\n\n".join(parts)

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": content
                            }
                        ]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error getting overview: {str(e)}"
                            }
                        ],
                        "isError": True
                    }
                }

        elif tool_name == "get_neighboring_chunks":
            source = arguments.get("source", "")
            chunk_index = arguments.get("chunk_index")
            context = arguments.get("context", 2)

            try:
                vs = get_vector_store()
                result = vs.get_neighboring_chunks(source, chunk_index, context=context)

                if "error" in result:
                    content = result["error"]
                else:
                    parts = []
                    for c in result["chunks"]:
                        marker = " ← TARGET" if c["is_target"] else ""
                        parts.append(f"**Chunk {c['index']}{marker}**\n{c['text']}\n")
                    content = "\n".join(parts)

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": content
                            }
                        ]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error getting neighboring chunks: {str(e)}"
                            }
                        ],
                        "isError": True
                    }
                }

        elif tool_name == "multi_search":
            query = arguments.get("query", "")
            queries = arguments.get("queries")
            num_results = arguments.get("num_results", 10)
            source_filter = arguments.get("source_filter")

            try:
                vs = get_vector_store()

                # If no additional queries provided, auto-generate variations
                search_queries = queries if queries else [query]

                # Execute all searches
                all_results = {}
                for q in search_queries:
                    results = vs.search(q, k=num_results * 2, source_filter=source_filter)
                    for r in results:
                        doc_id = r["metadata"].get("source", "?") + "_" + str(r["metadata"].get("chunk_index", "?"))
                        if doc_id not in all_results:
                            all_results[doc_id] = r

                # Sort by distance (best matches first)
                sorted_results = sorted(all_results.values(), key=lambda x: x["distance"])[:num_results]

                if not sorted_results:
                    content = f"No results found for any of the queries."
                else:
                    content_parts = []
                    for i, r in enumerate(sorted_results, 1):
                        relevance = 1 - r["distance"]
                        source = r["metadata"].get("source", "unknown")
                        text = r["text"]
                        content_parts.append(
                            f"**Result {i}** (Relevance: {relevance:.0%}, Source: {source})\n{text}"
                        )
                    content = "\n\n---\n\n".join(content_parts)
                    content = f"Searched {len(search_queries)} queries, found {len(sorted_results)} unique results:\n\n" + content

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": content
                            }
                        ]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error in multi-search: {str(e)}"
                            }
                        ],
                        "isError": True
                    }
                }

        elif tool_name == "replace_source":
            source = arguments.get("source", "")
            content_type = arguments.get("content_type", "")
            content = arguments.get("content", "")
            use_semantic = arguments.get("use_semantic_chunking", True)

            try:
                vs = get_vector_store()

                # Delete existing source
                deleted = vs.delete_by_source(source)

                # Extract text based on content_type
                text = ""
                if content_type == "file":
                    from pathlib import Path as P
                    fp = P(content)
                    if not fp.exists():
                        raise FileNotFoundError(f"File not found: {content}")
                    ext = fp.suffix.lower()
                    if ext == ".pdf":
                        from src.pdf_processor import extract_text_from_pdf
                        text = extract_text_from_pdf(content)
                    elif ext == ".epub":
                        from src.epub_processor import extract_text_from_epub
                        text = extract_text_from_epub(content)
                    elif ext == ".txt":
                        text = fp.read_text(encoding="utf-8")
                    else:
                        raise ValueError(f"Unsupported file type: {ext}")

                elif content_type == "url":
                    import requests
                    from bs4 import BeautifulSoup
                    response = requests.get(content, timeout=30, headers={
                        "User-Agent": "Mozilla/5.0 (compatible; StartupKnowledgeRAG/1.0)"
                    })
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, "html.parser")
                    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                        tag.decompose()
                    text = soup.get_text(separator="\n", strip=True)

                elif content_type == "text":
                    text = content

                else:
                    raise ValueError(f"Unknown content_type: {content_type}")

                if not text.strip():
                    raise ValueError("No content to ingest.")

                # Chunk and add new content
                from src.chunker import create_chunks_with_metadata

                chunks = create_chunks_with_metadata(
                    text=text,
                    source=source,
                    use_semantic=use_semantic,
                )

                texts = [c["text"] for c in chunks]
                metadatas = [c["metadata"] for c in chunks]
                ids = [f"{source}_{c['metadata']['chunk_index']}" for c in chunks]

                vs.add_documents(texts=texts, metadatas=metadatas, ids=ids)

                stats = vs.get_stats()
                content_msg = (
                    f"Successfully replaced source '{source}'.\n"
                    f"Deleted {deleted} old chunks, added {len(chunks)} new chunks.\n"
                    f"Knowledge base now contains {stats['total_documents']} total chunks."
                )

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": content_msg
                            }
                        ]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error replacing source: {str(e)}"
                            }
                        ],
                        "isError": True
                    }
                }

        elif tool_name == "batch_ingest_directory":
            directory = arguments.get("directory", "")
            use_semantic = arguments.get("use_semantic_chunking", True)

            try:
                from pathlib import Path as P
                import os

                dir_path = P(directory)
                if not dir_path.exists() or not dir_path.is_dir():
                    raise ValueError(f"Directory not found or not a directory: {directory}")

                # Find all supported files
                supported_exts = {".pdf", ".epub", ".txt"}
                files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in supported_exts]

                if not files:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"No supported files found in {directory}"
                                }
                            ]
                        }
                    }

                vs = get_vector_store()
                results = {"success": [], "error": []}

                for file_path in sorted(files):
                    try:
                        source = file_path.stem
                        ext = file_path.suffix.lower()

                        if ext == ".pdf":
                            from src.pdf_processor import extract_text_from_pdf
                            text = extract_text_from_pdf(str(file_path))
                        elif ext == ".epub":
                            from src.epub_processor import extract_text_from_epub
                            text = extract_text_from_epub(str(file_path))
                        elif ext == ".txt":
                            text = file_path.read_text(encoding="utf-8")

                        from src.chunker import create_chunks_with_metadata

                        chunks = create_chunks_with_metadata(
                            text=text,
                            source=source,
                            use_semantic=use_semantic,
                        )

                        texts = [c["text"] for c in chunks]
                        metadatas = [c["metadata"] for c in chunks]
                        ids = [f"{source}_{c['metadata']['chunk_index']}" for c in chunks]

                        vs.add_documents(texts=texts, metadatas=metadatas, ids=ids)
                        results["success"].append(f"{file_path.name} ({len(chunks)} chunks)")

                    except Exception as e:
                        results["error"].append(f"{file_path.name}: {str(e)}")

                stats = vs.get_stats()
                summary = f"Processed {len(files)} files:\n"
                if results["success"]:
                    summary += f"\n**Success ({len(results['success'])}):**\n" + "\n".join([f"- {s}" for s in results["success"]])
                if results["error"]:
                    summary += f"\n\n**Errors ({len(results['error'])}):**\n" + "\n".join([f"- {e}" for e in results["error"]])
                summary += f"\n\nKnowledge base now contains {stats['total_documents']} total chunks."

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": summary
                            }
                        ]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error in batch ingest: {str(e)}"
                            }
                        ],
                        "isError": True
                    }
                }

        # ── Tool 1: keyword_search ──
        elif tool_name == "keyword_search":
            keyword = arguments.get("keyword", "")
            num_results = arguments.get("num_results", 10)
            source_filter = arguments.get("source_filter")

            try:
                vs = get_vector_store()
                results = vs.keyword_search(keyword, k=num_results, source_filter=source_filter)

                if not results:
                    content = f"No chunks contain the keyword '{keyword}'."
                else:
                    parts = []
                    for i, r in enumerate(results, 1):
                        src = r["metadata"].get("source", "?")
                        idx = r["metadata"].get("chunk_index", "?")
                        parts.append(f"**Match {i}** (Source: {src}, Chunk {idx})\n{r['text']}")
                    content = f"Found {len(results)} chunk(s) containing '{keyword}':\n\n" + "\n\n---\n\n".join(parts)

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": content}]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error in keyword search: {str(e)}"}],
                        "isError": True
                    }
                }

        # ── Tool 2: export_source ──
        elif tool_name == "export_source":
            source = arguments.get("source", "")

            try:
                vs = get_vector_store()
                text = vs.export_source(source)

                if not text:
                    content = f"No content found for source '{source}'."
                else:
                    content = f"**Exported source '{source}'** ({len(text)} characters):\n\n{text}"

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": content}]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error exporting source: {str(e)}"}],
                        "isError": True
                    }
                }

        # ── Tool 3: tag_source ──
        elif tool_name == "tag_source":
            source = arguments.get("source", "")
            tags = arguments.get("tags", [])
            action = arguments.get("action", "add")

            try:
                meta = _load_meta()
                current_tags = meta.get("tags", {}).get(source, [])

                if action == "add":
                    new_tags = list(set(current_tags + tags))
                elif action == "remove":
                    new_tags = [t for t in current_tags if t not in tags]
                elif action == "set":
                    new_tags = list(set(tags))
                else:
                    raise ValueError(f"Unknown action: {action}. Use 'add', 'remove', or 'set'.")

                meta.setdefault("tags", {})[source] = new_tags
                _save_meta(meta)

                content = f"Tags for '{source}': {new_tags}"

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": content}]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error tagging source: {str(e)}"}],
                        "isError": True
                    }
                }

        # ── Tool 4: search_by_tag ──
        elif tool_name == "search_by_tag":
            query = arguments.get("query", "")
            tags = arguments.get("tags", [])
            num_results = arguments.get("num_results", 5)

            try:
                meta = _load_meta()
                tag_data = meta.get("tags", {})

                # Find sources that have ALL requested tags
                matching_sources = []
                for src, src_tags in tag_data.items():
                    if all(t in src_tags for t in tags):
                        matching_sources.append(src)

                if not matching_sources:
                    content = f"No sources found with tags {tags}."
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": content}]
                        }
                    }

                vs = get_vector_store()
                results = vs.search(query, k=num_results, source_filter=matching_sources)

                if not results:
                    content = f"No results found for query in sources with tags {tags}."
                else:
                    parts = []
                    for i, r in enumerate(results, 1):
                        relevance = 1 - r["distance"]
                        src = r["metadata"].get("source", "?")
                        parts.append(f"**Result {i}** (Relevance: {relevance:.0%}, Source: {src})\n{r['text']}")
                    content = f"Searched {len(matching_sources)} source(s) with tags {tags}:\n\n" + "\n\n---\n\n".join(parts)

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": content}]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error in tag search: {str(e)}"}],
                        "isError": True
                    }
                }

        # ── Tool 5: compare_sources ──
        elif tool_name == "compare_sources":
            topic = arguments.get("topic", "")
            source_a = arguments.get("source_a", "")
            source_b = arguments.get("source_b", "")
            num_results = arguments.get("num_results", 3)

            try:
                vs = get_vector_store()
                results_a = vs.search(topic, k=num_results, source_filter=source_a)
                results_b = vs.search(topic, k=num_results, source_filter=source_b)

                parts = [f"## Comparing: \"{topic}\"\n"]

                parts.append(f"### Source A: {source_a}")
                if results_a:
                    for i, r in enumerate(results_a, 1):
                        rel = 1 - r["distance"]
                        parts.append(f"**{i}.** (Relevance: {rel:.0%})\n{r['text']}")
                else:
                    parts.append("_No results found._")

                parts.append(f"\n### Source B: {source_b}")
                if results_b:
                    for i, r in enumerate(results_b, 1):
                        rel = 1 - r["distance"]
                        parts.append(f"**{i}.** (Relevance: {rel:.0%})\n{r['text']}")
                else:
                    parts.append("_No results found._")

                content = "\n\n".join(parts)

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": content}]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error comparing sources: {str(e)}"}],
                        "isError": True
                    }
                }

        # ── Tool 6: chunk_quality_report ──
        elif tool_name == "chunk_quality_report":
            source = arguments.get("source")

            try:
                vs = get_vector_store()
                report = vs.get_quality_report(source=source)

                if "error" in report:
                    content = report["error"]
                else:
                    scope = f"source '{source}'" if source else "entire knowledge base"
                    lines = [
                        f"## Chunk Quality Report ({scope})\n",
                        f"- **Total chunks:** {report['total_chunks']}",
                        f"- **Avg length:** {report['avg_length']:.0f} chars",
                        f"- **Min length:** {report['min_length']} chars",
                        f"- **Max length:** {report['max_length']} chars",
                        f"- **Short chunks (<50 chars):** {report['short_chunks_count']}",
                        f"- **Long chunks (>2000 chars):** {report['long_chunks_count']}",
                        f"- **Whitespace-heavy (>50%):** {report['whitespace_heavy_count']}",
                    ]
                    if report["short_chunks"]:
                        lines.append("\n**Flagged short chunks:**")
                        for c in report["short_chunks"]:
                            lines.append(f"  - {c['source']} chunk {c['chunk_index']}: {c['length']} chars")
                    if report["long_chunks"]:
                        lines.append("\n**Flagged long chunks:**")
                        for c in report["long_chunks"]:
                            lines.append(f"  - {c['source']} chunk {c['chunk_index']}: {c['length']} chars")
                    if report["whitespace_heavy"]:
                        lines.append("\n**Flagged whitespace-heavy chunks:**")
                        for c in report["whitespace_heavy"]:
                            lines.append(f"  - {c['source']} chunk {c['chunk_index']}: {c['length']} chars")
                    content = "\n".join(lines)

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": content}]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error generating quality report: {str(e)}"}],
                        "isError": True
                    }
                }

        # ── Tool 7: reindex_source ──
        elif tool_name == "reindex_source":
            source = arguments.get("source", "")
            use_semantic = arguments.get("use_semantic_chunking", True)
            chunk_size = arguments.get("chunk_size", 1000)
            chunk_overlap = arguments.get("chunk_overlap", 200)

            try:
                vs = get_vector_store()

                # Export existing text (note: original chunk boundaries are lost on re-join)
                text = vs.export_source(source)
                if not text:
                    raise ValueError(f"No content found for source '{source}'.")

                # Delete old chunks
                deleted = vs.delete_by_source(source)

                # Re-chunk with new settings
                from src.chunker import create_chunks_with_metadata
                chunks = create_chunks_with_metadata(
                    text=text,
                    source=source,
                    use_semantic=use_semantic,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                texts = [c["text"] for c in chunks]
                metadatas = [c["metadata"] for c in chunks]
                ids = [f"{source}_{c['metadata']['chunk_index']}" for c in chunks]

                vs.add_documents(texts=texts, metadatas=metadatas, ids=ids)

                content = (
                    f"Reindexed source '{source}'.\n"
                    f"Deleted {deleted} old chunks, created {len(chunks)} new chunks.\n"
                    f"Settings: semantic={use_semantic}, chunk_size={chunk_size}, overlap={chunk_overlap}"
                )

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": content}]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error reindexing source: {str(e)}"}],
                        "isError": True
                    }
                }

        # ── Tool 8: annotate_chunk ──
        elif tool_name == "annotate_chunk":
            source = arguments.get("source", "")
            chunk_index = arguments.get("chunk_index", 0)
            note = arguments.get("note")

            try:
                from datetime import datetime, timezone
                meta = _load_meta()
                key = f"{source}_{chunk_index}"
                annotations = meta.get("annotations", {})

                if note is None:
                    # View mode
                    existing = annotations.get(key, [])
                    if not existing:
                        content = f"No annotations for {source} chunk {chunk_index}."
                    else:
                        parts = [f"**Annotations for {source} chunk {chunk_index}:**\n"]
                        for a in existing:
                            parts.append(f"- [{a.get('created', '?')}] {a.get('note', '')}")
                        content = "\n".join(parts)
                else:
                    # Add mode
                    annotations.setdefault(key, []).append({
                        "note": note,
                        "created": datetime.now(timezone.utc).isoformat()
                    })
                    meta["annotations"] = annotations
                    _save_meta(meta)
                    content = f"Annotation added to {source} chunk {chunk_index}. Total annotations: {len(annotations[key])}"

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": content}]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error annotating chunk: {str(e)}"}],
                        "isError": True
                    }
                }

        # ── Tool 9: random_chunk ──
        elif tool_name == "random_chunk":
            num_chunks = arguments.get("num_chunks", 1)
            source = arguments.get("source")

            try:
                vs = get_vector_store()
                chunks = vs.get_random_chunks(n=num_chunks, source=source)

                if not chunks:
                    content = "No chunks found."
                else:
                    parts = []
                    for i, c in enumerate(chunks, 1):
                        src = c["metadata"].get("source", "?")
                        idx = c["metadata"].get("chunk_index", "?")
                        parts.append(f"**Random {i}** (Source: {src}, Chunk {idx})\n{c['text']}")
                    content = "\n\n---\n\n".join(parts)

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": content}]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error getting random chunks: {str(e)}"}],
                        "isError": True
                    }
                }

        # ── Tool 10: semantic_clusters ──
        elif tool_name == "semantic_clusters":
            num_clusters = arguments.get("num_clusters", 5)
            source = arguments.get("source")

            try:
                vs = get_vector_store()
                result = vs.get_semantic_clusters(n_clusters=num_clusters, source=source)

                if "error" in result:
                    content = result["error"]
                else:
                    parts = [f"## Semantic Clusters ({result['total_chunks']} chunks → {result['n_clusters']} clusters)\n"]
                    for name, cluster in result["clusters"].items():
                        rep = cluster["representative"]
                        parts.append(
                            f"### {name} ({cluster['size']} chunks)\n"
                            f"**Representative** (Source: {rep['source']}, Chunk {rep['chunk_index']}):\n"
                            f"_{rep['text']}_\n"
                        )
                        if cluster["members"]:
                            parts.append("**Members:**")
                            for m in cluster["members"]:
                                parts.append(f"- {m['source']} chunk {m['chunk_index']}: _{m['preview']}_...")
                    content = "\n\n".join(parts)

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": content}]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error computing clusters: {str(e)}"}],
                        "isError": True
                    }
                }

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}"
                }
            }

    elif method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {}
        }

    else:
        # Unknown methods - return empty result instead of error for robustness
        if req_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
        return None  # Ignore unknown notifications


def main():
    """Main loop - read from stdin, write to stdout."""
    sys.stderr.write("MCP server starting...\n")
    sys.stderr.flush()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            line = line.strip()
            if not line:
                continue

            request = json.loads(line)
            response = handle_request(request)

            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                }
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"Error: {str(e)}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
