#!/usr/bin/env python3
"""
CLI skripta za dodavanje PDF-ova u knowledge base.

Usage:
    python scripts/ingest_pdf.py --pdf data/raw/knjiga.pdf
    python scripts/ingest_pdf.py --dir data/raw/
"""

import argparse
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_processor import extract_text_from_pdf, process_multiple_pdfs
from src.chunker import create_chunks_with_metadata
from src.vector_store import VectorStore


def ingest_single_pdf(pdf_path: str, vector_store: VectorStore, chunk_size: int = 1000, chunk_overlap: int = 200, use_semantic: bool = True):
    """Ingestuje jedan PDF."""
    chunking_method = "semantic" if use_semantic else "character-based"
    print(f"[PDF] Procesiram: {pdf_path}")
    print(f"      Metoda: {chunking_method}")

    # Extract text
    text = extract_text_from_pdf(pdf_path)
    print(f"      Ekstrahirano: {len(text)} karaktera")

    # Create chunks
    chunks = create_chunks_with_metadata(
        text,
        source=Path(pdf_path).name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_semantic=use_semantic
    )
    print(f"      Kreirano: {len(chunks)} chunk-ova ({chunking_method})")

    # Add to vector store
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"{Path(pdf_path).name}_{i}" for i in range(len(chunks))]

    vector_store.add_documents(texts, metadatas, ids)
    print(f"      [OK] Dodato u bazu!")

    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF-ove u knowledge base")
    parser.add_argument("--pdf", type=str, help="Putanja do PDF fajla")
    parser.add_argument("--dir", type=str, help="Putanja do direktorijuma sa PDF-ovima")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Velicina chunk-a (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap chunk-ova (default: 200)")
    parser.add_argument("--db-path", type=str, default="./chroma_db", help="Putanja do ChromaDB")
    parser.add_argument("--semantic", action="store_true", default=True, help="Koristi semantic chunking (default: True)")
    parser.add_argument("--no-semantic", action="store_false", dest="semantic", help="Koristi character-based chunking")

    args = parser.parse_args()

    if not args.pdf and not args.dir:
        parser.print_help()
        sys.exit(1)

    # Initialize vector store
    vector_store = VectorStore(persist_directory=args.db_path)
    print(f"[STATS] Trenutno u bazi: {vector_store.get_stats()['total_documents']} chunk-ova\n")

    total_chunks = 0

    if args.pdf:
        if not Path(args.pdf).exists():
            print(f"[ERROR] Fajl ne postoji: {args.pdf}")
            sys.exit(1)
        total_chunks = ingest_single_pdf(args.pdf, vector_store, args.chunk_size, args.chunk_overlap, args.semantic)

    if args.dir:
        pdf_dir = Path(args.dir)
        if not pdf_dir.exists():
            print(f"[ERROR] Direktorijum ne postoji: {args.dir}")
            sys.exit(1)

        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"[WARN] Nema PDF fajlova u: {args.dir}")
            sys.exit(0)

        print(f"[DIR] Pronadeno {len(pdf_files)} PDF fajlova\n")

        for pdf_file in pdf_files:
            chunks = ingest_single_pdf(str(pdf_file), vector_store, args.chunk_size, args.chunk_overlap, args.semantic)
            total_chunks += chunks
            print()

    print("=" * 50)
    print(f"[DONE] Zavrseno! Ukupno dodato: {total_chunks} chunk-ova")
    print(f"[STATS] Ukupno u bazi: {vector_store.get_stats()['total_documents']} chunk-ova")


if __name__ == "__main__":
    main()
