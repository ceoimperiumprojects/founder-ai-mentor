#!/usr/bin/env python3
"""
CLI skripta za dodavanje EPUB-ova u knowledge base.

Usage:
    python scripts/ingest_epub.py --epub knjiga.epub
"""

import argparse
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.epub_processor import extract_text_from_epub
from src.chunker import create_chunks_with_metadata
from src.vector_store import VectorStore


def ingest_single_epub(epub_path: str, vector_store: VectorStore, use_semantic: bool = True):
    """Ingestuje jedan EPUB."""
    chunking_method = "semantic" if use_semantic else "character-based"
    print(f"[EPUB] Procesiram: {epub_path}")
    print(f"       Metoda: {chunking_method}")

    # Extract text
    text = extract_text_from_epub(epub_path)
    print(f"       Ekstrahirano: {len(text)} karaktera")

    # Create chunks
    chunks = create_chunks_with_metadata(
        text,
        source=Path(epub_path).name,
        use_semantic=use_semantic
    )
    print(f"       Kreirano: {len(chunks)} chunk-ova ({chunking_method})")

    # Add to vector store
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"{Path(epub_path).name}_{i}" for i in range(len(chunks))]

    vector_store.add_documents(texts, metadatas, ids)
    print(f"       [OK] Dodato u bazu!")

    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest EPUB-ove u knowledge base")
    parser.add_argument("--epub", type=str, required=True, help="Putanja do EPUB fajla")
    parser.add_argument("--db-path", type=str, default="./chroma_db", help="Putanja do ChromaDB")
    parser.add_argument("--semantic", action="store_true", default=True, help="Koristi semantic chunking (default: True)")
    parser.add_argument("--no-semantic", action="store_false", dest="semantic", help="Koristi character-based chunking")

    args = parser.parse_args()

    if not Path(args.epub).exists():
        print(f"[ERROR] Fajl ne postoji: {args.epub}")
        sys.exit(1)

    # Initialize vector store
    vector_store = VectorStore(persist_directory=args.db_path)
    print(f"[STATS] Trenutno u bazi: {vector_store.get_stats()['total_documents']} chunk-ova\n")

    total_chunks = ingest_single_epub(args.epub, vector_store, args.semantic)

    print("=" * 50)
    print(f"[DONE] Zavrseno! Ukupno dodato: {total_chunks} chunk-ova")
    print(f"[STATS] Ukupno u bazi: {vector_store.get_stats()['total_documents']} chunk-ova")


if __name__ == "__main__":
    main()
