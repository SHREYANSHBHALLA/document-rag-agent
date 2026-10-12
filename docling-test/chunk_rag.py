from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter


# 1. Read parsed Markdown
with open("parsed_rag.md", "r", encoding="utf-8") as f:
    markdown = f.read()


# 2. Split document by headings first
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,
)

sections = header_splitter.split_text(markdown)

print("Total sections:", len(sections))


# 3. Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Make SentenceTransformer compatible with LangChain
class SentenceTransformerEmbeddings(Embeddings):

    def embed_documents(self, texts):
        return model.encode(texts).tolist()

    def embed_query(self, text):
        return model.encode(text).tolist()


embeddings = SentenceTransformerEmbeddings()


# 5. Create LangChain SemanticChunker
semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90,
)


# 6. Apply semantic chunking to every section
semantic_chunks = []

for section in sections:

    chunks = semantic_splitter.split_text(
        section.page_content
    )

    for chunk in chunks:

        semantic_chunks.append({
            "text": chunk,
            "metadata": section.metadata
        })


# 7. Display results
print("\nTotal semantic chunks:", len(semantic_chunks))


for i, chunk in enumerate(semantic_chunks[:10]):

    print(f"\n{'=' * 60}")
    print(f"CHUNK {i + 1}")
    print(f"{'=' * 60}")

    print(chunk["text"][:1500])

    print("\nMetadata:")
    print(chunk["metadata"])