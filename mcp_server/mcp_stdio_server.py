#!/usr/bin/env python3
"""
MCP Server za Startup Knowledge RAG - stdio verzija za Claude Desktop.
"""

import sys
import json
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import VectorStore

# Initialize vector store
project_root = Path(__file__).parent.parent
db_path = project_root / "chroma_db"

# Warmup: Load model at startup instead of first request
sys.stderr.write("Loading embedding model...\n")
sys.stderr.flush()
vector_store = VectorStore(persist_directory=str(db_path))
# Do a dummy search to fully warm up the model
_ = vector_store.search("warmup", k=1)
sys.stderr.write("Model loaded and ready!\n")
sys.stderr.flush()


def get_vector_store():
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
                "protocolVersion": "2024-11-05",
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

            try:
                vs = get_vector_store()
                results = vs.search(query, k=num_results)

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

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}"
                }
            }

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}"
            }
        }


def main():
    """Main loop - read from stdin, write to stdout."""
    # Disable buffering for stderr (for logging)
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

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


if __name__ == "__main__":
    main()
