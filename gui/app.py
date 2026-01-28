import streamlit as st
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_processor import extract_text_from_pdf
from src.chunker import create_chunks_with_metadata
from src.vector_store import VectorStore

st.set_page_config(
    page_title="Startup Knowledge Manager",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Startup Knowledge Manager")
st.markdown("Dodaj znanje u svoju AI bazu podataka")

# Initialize vector store
@st.cache_resource
def get_vector_store():
    project_root = Path(__file__).parent.parent
    db_path = project_root / "chroma_db"
    return VectorStore(persist_directory=str(db_path))

vector_store = get_vector_store()

# Sidebar stats
stats = vector_store.get_stats()
st.sidebar.header("📊 Statistika")
st.sidebar.metric("Ukupno chunk-ova", stats["total_documents"])

st.sidebar.divider()
st.sidebar.header("⚙️ Postavke")
chunk_size = st.sidebar.slider("Chunk size", 500, 2000, 1000, 100)
chunk_overlap = st.sidebar.slider("Chunk overlap", 50, 500, 200, 50)

# Main content
tab1, tab2 = st.tabs(["📤 Dodaj znanje", "🔍 Pretraga"])

with tab1:
    st.header("Upload PDF")

    uploaded_file = st.file_uploader(
        "Izaberi PDF fajl",
        type=["pdf"],
        help="Dodaj PDF fajl sa znanjem koje zelis da AI zapamti"
    )

    if uploaded_file:
        with st.spinner("Procesiram PDF..."):
            # Save temporarily
            temp_dir = Path("./temp")
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / uploaded_file.name

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract text
            text = extract_text_from_pdf(str(temp_path))

            # Chunk
            chunks = create_chunks_with_metadata(
                text,
                source=uploaded_file.name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            # Clean up temp file
            temp_path.unlink()

        # Preview
        st.success(f"Pronadeno **{len(chunks)}** chunk-ova")

        with st.expander("Pregledaj chunk-ove", expanded=False):
            for i, chunk in enumerate(chunks[:5]):
                st.text_area(
                    f"Chunk {i+1}",
                    chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"],
                    height=100,
                    key=f"chunk_{i}"
                )
            if len(chunks) > 5:
                st.info(f"... i jos {len(chunks) - 5} chunk-ova")

        # Add to database
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Dodaj u bazu znanja", type="primary", use_container_width=True):
                with st.spinner("Dodajem u bazu..."):
                    texts = [c["text"] for c in chunks]
                    metadatas = [c["metadata"] for c in chunks]
                    ids = [f"{uploaded_file.name}_{i}" for i in range(len(chunks))]

                    vector_store.add_documents(texts, metadatas, ids)

                st.success(f"Uspjesno dodato {len(chunks)} chunk-ova!")
                st.cache_resource.clear()
                st.rerun()

with tab2:
    st.header("Testiranje pretrage")

    query = st.text_input(
        "Postavi pitanje",
        placeholder="Kako da validam startup ideju?"
    )

    num_results = st.slider("Broj rezultata", 1, 10, 5)

    if query:
        with st.spinner("Pretrazujem..."):
            results = vector_store.search(query, k=num_results)

        if results:
            for i, result in enumerate(results):
                relevance = 1 - result["distance"]

                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**Rezultat {i+1}**")
                    with col2:
                        st.metric("Relevantnost", f"{relevance:.0%}")

                    st.text_area(
                        "Tekst",
                        result["text"],
                        height=150,
                        key=f"result_{i}",
                        label_visibility="collapsed"
                    )
                    st.caption(f"Izvor: {result['metadata'].get('source', 'N/A')}")
                    st.divider()
        else:
            st.warning("Nema rezultata. Dodaj prvo znanje u bazu!")

# Footer
st.sidebar.divider()
st.sidebar.markdown("---")
st.sidebar.markdown("**Startup Knowledge RAG** v1.0.0")
st.sidebar.markdown("by Pavle Andjelkovic")
