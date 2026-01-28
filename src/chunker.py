from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list[str]:
    """Dijeli tekst na chunk-ove."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks


def create_chunks_with_metadata(
    text: str,
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list[dict]:
    """Kreira chunk-ove sa metadata."""
    chunks = create_chunks(text, chunk_size, chunk_overlap)
    return [
        {
            "text": chunk,
            "metadata": {
                "source": source,
                "chunk_index": i
            }
        }
        for i, chunk in enumerate(chunks)
    ]


if __name__ == "__main__":
    sample_text = "Ovo je test tekst. " * 100
    chunks = create_chunks(sample_text)
    print(f"Created {len(chunks)} chunks")
