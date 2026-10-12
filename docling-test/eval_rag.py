import json
import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from langchain_core.embeddings import Embeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from rank_bm25 import BM25Okapi


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).parent

SOURCE_FILE = BASE_DIR / "parsed_rag.md"
GOLDEN_FILE = BASE_DIR / "golden_dataset.json"
RESULTS_FILE = BASE_DIR / "evaluation_results.txt"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_CANDIDATE_K = 15
BM25_CANDIDATE_K = 15
FINAL_K = 5

RECURSIVE_CHUNK_SIZE = 800
RECURSIVE_CHUNK_OVERLAP = 100

SEMANTIC_BREAKPOINT_TYPE = "percentile"
SEMANTIC_BREAKPOINT_AMOUNT = 90

RRF_CONSTANT = 60

GOLD_TOKEN_RECALL_THRESHOLD = 0.70


# ============================================================
# Markdown section splitting
# ============================================================

def split_markdown_with_hierarchy(text):

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )

    return splitter.split_text(text)


# ============================================================
# Metadata
# ============================================================

def add_metadata(chunks):

    for chunk in chunks:

        section_values = [
            chunk.metadata.get("h1", ""),
            chunk.metadata.get("h2", ""),
            chunk.metadata.get("h3", ""),
        ]

        section_text = " ".join(
            section_values
        ).upper()

        chunk.metadata["is_reference"] = (
            "REFERENCES" in section_text
        )

    return chunks


def filter_chunks(chunks):

    return [
        chunk
        for chunk in chunks
        if not chunk.metadata.get(
            "is_reference",
            False,
        )
    ]


# ============================================================
# Recursive chunking
# ============================================================

def build_recursive_chunks(sections):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for section in sections:

        section_chunks = splitter.split_documents(
            [section]
        )

        for chunk in section_chunks:

            chunk.metadata = dict(
                section.metadata
            )

            chunks.append(chunk)

    return add_metadata(chunks)


# ============================================================
# Embedding wrapper
# ============================================================

class SentenceTransformerEmbeddings(Embeddings):

    def __init__(self, model_name):

        self.model = SentenceTransformer(
            model_name
        )

    def embed_documents(self, texts):

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def embed_query(self, text):

        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        return embedding.tolist()


# ============================================================
# Semantic chunking
# ============================================================

def build_semantic_chunks(
    sections,
    embeddings,
):

    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=(
            SEMANTIC_BREAKPOINT_TYPE
        ),
        breakpoint_threshold_amount=(
            SEMANTIC_BREAKPOINT_AMOUNT
        ),
    )

    chunks = []

    for section in sections:

        section_chunks = (
            semantic_splitter.split_documents(
                [section]
            )
        )

        for chunk in section_chunks:

            chunk.metadata = dict(
                section.metadata
            )

            chunks.append(chunk)

    return add_metadata(chunks)


# ============================================================
# BM25
# ============================================================

def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


def build_bm25_index(chunks):

    tokenized_documents = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]

    return BM25Okapi(
        tokenized_documents
    )


def retrieve_bm25(
    question,
    chunks,
    bm25_index,
    k,
):

    start = time.perf_counter()

    query_tokens = tokenize(question)

    scores = bm25_index.get_scores(
        query_tokens
    )

    ranked_indices = np.argsort(
        scores
    )[::-1][:k]

    results = []

    for rank, index in enumerate(
        ranked_indices,
        start=1,
    ):

        index = int(index)

        results.append(
            {
                "index": index,
                "rank": rank,
                "score": float(
                    scores[index]
                ),
                "chunk": chunks[index],
            }
        )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return results, elapsed_ms


# ============================================================
# Vector retrieval
# ============================================================

def create_embeddings(
    chunks,
    embedding_model,
):

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    return np.array(
        embedding_model.embed_documents(
            texts
        ),
        dtype=np.float32,
    )


def retrieve_vector(
    question,
    chunks,
    chunk_embeddings,
    embedding_model,
    k,
):

    start = time.perf_counter()

    query_embedding = np.array(
        embedding_model.embed_query(
            question
        ),
        dtype=np.float32,
    )

    # Embeddings are normalized,
    # therefore dot product = cosine similarity.

    scores = (
        chunk_embeddings
        @ query_embedding
    )

    ranked_indices = np.argsort(
        scores
    )[::-1][:k]

    results = []

    for rank, index in enumerate(
        ranked_indices,
        start=1,
    ):

        index = int(index)

        results.append(
            {
                "index": index,
                "rank": rank,
                "score": float(
                    scores[index]
                ),
                "chunk": chunks[index],
            }
        )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return results, elapsed_ms


# ============================================================
# Reciprocal Rank Fusion
# ============================================================

def reciprocal_rank_fusion(
    vector_results,
    bm25_results,
):

    start = time.perf_counter()

    fused = {}

    def add_results(
        results,
        source,
    ):

        for result in results:

            index = result["index"]
            rank = result["rank"]

            if index not in fused:

                fused[index] = {
                    "index": index,
                    "rrf_score": 0.0,
                    "sources": [],
                    "vector_rank": None,
                    "bm25_rank": None,
                    "vector_score": None,
                    "bm25_score": None,
                    "chunk": result["chunk"],
                }

            fused[index]["rrf_score"] += (
                1.0
                / (
                    RRF_CONSTANT
                    + rank
                )
            )

            if source not in fused[index][
                "sources"
            ]:

                fused[index]["sources"].append(
                    source
                )

            if source == "vector":

                fused[index]["vector_rank"] = (
                    rank
                )

                fused[index]["vector_score"] = (
                    result["score"]
                )

            elif source == "bm25":

                fused[index]["bm25_rank"] = (
                    rank
                )

                fused[index]["bm25_score"] = (
                    result["score"]
                )

    add_results(
        vector_results,
        "vector",
    )

    add_results(
        bm25_results,
        "bm25",
    )

    results = sorted(
        fused.values(),
        key=lambda item: item[
            "rrf_score"
        ],
        reverse=True,
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return results, elapsed_ms


# ============================================================
# Gold passage matching
# ============================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def gold_passage_is_relevant(
    chunk_text,
    gold_passage,
):

    chunk_tokens = tokenize(
        chunk_text
    )

    gold_tokens = tokenize(
        gold_passage
    )

    if not gold_tokens:
        return False

    chunk_token_set = set(
        chunk_tokens
    )

    matched_tokens = sum(
        1
        for token in gold_tokens
        if token in chunk_token_set
    )

    token_recall = (
        matched_tokens
        / len(gold_tokens)
    )

    normalized_chunk = (
        normalize_text(chunk_text)
    )

    normalized_gold = (
        normalize_text(gold_passage)
    )

    if normalized_gold in normalized_chunk:
        return True

    return (
        token_recall
        >= GOLD_TOKEN_RECALL_THRESHOLD
    )


def chunk_is_relevant(
    chunk,
    gold_passages,
):

    return any(
        gold_passage_is_relevant(
            chunk.page_content,
            gold_passage,
        )
        for gold_passage in gold_passages
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    retrieved_results,
    gold_passages,
):

    retrieved_results = (
        retrieved_results[:FINAL_K]
    )

    relevance = [
        chunk_is_relevant(
            result["chunk"],
            gold_passages,
        )
        for result in retrieved_results
    ]

    metrics = {}

    for k in [1, 3, 5]:

        top_k = relevance[:k]

        relevant_count = sum(
            top_k
        )

        # Hit-based Recall@K:
        # Did we retrieve at least one
        # relevant chunk in top K?
        metrics[
            f"Recall@{k}"
        ] = (
            1.0
            if relevant_count > 0
            else 0.0
        )

        metrics[
            f"Precision@{k}"
        ] = (
            relevant_count / k
        )

    reciprocal_rank = 0.0

    for rank, is_relevant in enumerate(
        relevance,
        start=1,
    ):

        if is_relevant:

            reciprocal_rank = (
                1.0 / rank
            )

            break

    metrics["MRR"] = (
        reciprocal_rank
    )

    return metrics


# ============================================================
# Metric storage
# ============================================================

CONFIG_NAMES = [

    "Recursive Vector",

    "Recursive Vector + Metadata",

    "Recursive Hybrid",

    "Recursive Hybrid + Metadata",

    "Semantic Vector",

    "Semantic Vector + Metadata",

    "Semantic Hybrid",

    "Semantic Hybrid + Metadata",
]


def initialize_metrics():

    return {
        config: {
            "Recall@1": [],
            "Precision@1": [],

            "Recall@3": [],
            "Precision@3": [],

            "Recall@5": [],
            "Precision@5": [],

            "MRR": [],

            "vector_time_ms": [],
            "bm25_time_ms": [],
            "rrf_time_ms": [],
            "total_retrieval_time_ms": [],
        }
        for config in CONFIG_NAMES
    }


def add_metrics(
    metric_store,
    config,
    metrics,
):

    for metric_name, value in (
        metrics.items()
    ):

        metric_store[
            config
        ][metric_name].append(
            value
        )


def average_metrics(
    metric_store,
):

    averages = {}

    for config, metrics in (
        metric_store.items()
    ):

        averages[config] = {}

        for metric_name, values in (
            metrics.items()
        ):

            if values:

                averages[
                    config
                ][metric_name] = (
                    sum(values)
                    / len(values)
                )

            else:

                averages[
                    config
                ][metric_name] = 0.0

    return averages


# ============================================================
# Result table
# ============================================================

def build_results_table(
    averages,
):

    lines = []

    lines.append(
        "FINAL RETRIEVAL EVALUATION RESULTS"
    )

    lines.append(
        "=" * 150
    )

    header = (
        f"{'Configuration':<40}"
        f"{'R@1':>8}"
        f"{'P@1':>8}"
        f"{'R@3':>8}"
        f"{'P@3':>8}"
        f"{'R@5':>8}"
        f"{'P@5':>8}"
        f"{'MRR':>8}"
        f"{'Vector':>12}"
        f"{'BM25':>12}"
        f"{'RRF':>12}"
        f"{'Total':>12}"
    )

    lines.append(header)

    lines.append(
        "-" * 150
    )

    for config in CONFIG_NAMES:

        m = averages[config]

        lines.append(
            f"{config:<40}"
            f"{m['Recall@1']:>8.3f}"
            f"{m['Precision@1']:>8.3f}"
            f"{m['Recall@3']:>8.3f}"
            f"{m['Precision@3']:>8.3f}"
            f"{m['Recall@5']:>8.3f}"
            f"{m['Precision@5']:>8.3f}"
            f"{m['MRR']:>8.3f}"
            f"{m['vector_time_ms']:>12.2f}"
            f"{m['bm25_time_ms']:>12.2f}"
            f"{m['rrf_time_ms']:>12.2f}"
            f"{m['total_retrieval_time_ms']:>12.2f}"
        )

    lines.append(
        "=" * 150
    )

    return "\n".join(lines)


# ============================================================
# Save results
# ============================================================

def save_results(
    output,
    recursive_count,
    recursive_filtered_count,
    semantic_count,
    semantic_filtered_count,
):

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "RAG RETRIEVAL EVALUATION\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            "Configuration\n"
        )

        f.write(
            f"Embedding model: "
            f"{EMBEDDING_MODEL}\n"
        )

        f.write(
            f"Vector candidate K: "
            f"{VECTOR_CANDIDATE_K}\n"
        )

        f.write(
            f"BM25 candidate K: "
            f"{BM25_CANDIDATE_K}\n"
        )

        f.write(
            f"Final K: "
            f"{FINAL_K}\n"
        )

        f.write(
            f"RRF constant: "
            f"{RRF_CONSTANT}\n"
        )

        f.write(
            f"Recursive chunk size: "
            f"{RECURSIVE_CHUNK_SIZE}\n"
        )

        f.write(
            f"Recursive overlap: "
            f"{RECURSIVE_CHUNK_OVERLAP}\n"
        )

        f.write(
            f"Semantic breakpoint: "
            f"{SEMANTIC_BREAKPOINT_AMOUNT}\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "Dataset Statistics\n"
        )

        f.write(
            "-" * 40
            + "\n"
        )

        f.write(
            f"Markdown sections: "
            f"{len(sections)}\n"
        )

        f.write(
            f"Recursive chunks: "
            f"{recursive_count}\n"
        )

        f.write(
            f"Recursive filtered chunks: "
            f"{recursive_filtered_count}\n"
        )

        f.write(
            f"Semantic chunks: "
            f"{semantic_count}\n"
        )

        f.write(
            f"Semantic filtered chunks: "
            f"{semantic_filtered_count}\n"
        )

        f.write(
            f"Golden questions: "
            f"{len(golden_dataset)}\n"
        )

        f.write(
            "\n"
        )

        f.write(
            output
        )

        f.write(
            "\n\n"
        )

        f.write(
            "Metric Definitions\n"
        )

        f.write(
            "-" * 40
            + "\n"
        )

        f.write(
            "Recall@K: whether at least one "
            "relevant chunk appears in top K.\n"
        )

        f.write(
            "Precision@K: proportion of the "
            "top K chunks marked relevant.\n"
        )

        f.write(
            "MRR: reciprocal rank of the first "
            "relevant chunk.\n"
        )

        f.write(
            "Vector/BM25/RRF/Total times are "
            "retrieval-stage timings in milliseconds.\n"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not SOURCE_FILE.exists():

        raise FileNotFoundError(
            f"Source file not found: "
            f"{SOURCE_FILE}"
        )

    if not GOLDEN_FILE.exists():

        raise FileNotFoundError(
            f"Golden dataset not found: "
            f"{GOLDEN_FILE}"
        )

    # --------------------------------------------------------
    # Load source
    # --------------------------------------------------------

    print(
        "Loading source document..."
    )

    source_text = SOURCE_FILE.read_text(
        encoding="utf-8"
    )

    with open(
        GOLDEN_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        golden_dataset = json.load(f)

    print(
        f"Golden questions: "
        f"{len(golden_dataset)}"
    )

    # --------------------------------------------------------
    # Split sections
    # --------------------------------------------------------

    print(
        "\nSplitting Markdown sections..."
    )

    sections = (
        split_markdown_with_hierarchy(
            source_text
        )
    )

    print(
        f"Markdown sections: "
        f"{len(sections)}"
    )

    # --------------------------------------------------------
    # Recursive chunks
    # --------------------------------------------------------

    print(
        "\nCreating recursive chunks..."
    )

    recursive_chunks = (
        build_recursive_chunks(
            sections
        )
    )

    print(
        f"Recursive chunks: "
        f"{len(recursive_chunks)}"
    )

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    print(
        f"\nLoading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    embedding_model = (
        SentenceTransformerEmbeddings(
            EMBEDDING_MODEL
        )
    )

    # --------------------------------------------------------
    # Semantic chunks
    # --------------------------------------------------------

    print(
        "\nCreating semantic chunks..."
    )

    semantic_chunks = (
        build_semantic_chunks(
            sections,
            embedding_model,
        )
    )

    print(
        f"Semantic chunks: "
        f"{len(semantic_chunks)}"
    )

    # --------------------------------------------------------
    # Metadata filtering
    # --------------------------------------------------------

    recursive_filtered_chunks = (
        filter_chunks(
            recursive_chunks
        )
    )

    semantic_filtered_chunks = (
        filter_chunks(
            semantic_chunks
        )
    )

    print(
        "\nMetadata filtering:"
    )

    print(
        f"Recursive filtered chunks: "
        f"{len(recursive_filtered_chunks)}"
    )

    print(
        f"Semantic filtered chunks: "
        f"{len(semantic_filtered_chunks)}"
    )

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    print(
        "\nCreating recursive embeddings..."
    )

    recursive_embeddings = (
        create_embeddings(
            recursive_chunks,
            embedding_model,
        )
    )

    print(
        "Creating recursive filtered embeddings..."
    )

    recursive_filtered_embeddings = (
        create_embeddings(
            recursive_filtered_chunks,
            embedding_model,
        )
    )

    print(
        "Creating semantic embeddings..."
    )

    semantic_embeddings = (
        create_embeddings(
            semantic_chunks,
            embedding_model,
        )
    )

    print(
        "Creating semantic filtered embeddings..."
    )

    semantic_filtered_embeddings = (
        create_embeddings(
            semantic_filtered_chunks,
            embedding_model,
        )
    )

    # --------------------------------------------------------
    # BM25 indexes
    # --------------------------------------------------------

    print(
        "\nBuilding BM25 indexes..."
    )

    recursive_bm25_index = (
        build_bm25_index(
            recursive_chunks
        )
    )

    recursive_filtered_bm25_index = (
        build_bm25_index(
            recursive_filtered_chunks
        )
    )

    semantic_bm25_index = (
        build_bm25_index(
            semantic_chunks
        )
    )

    semantic_filtered_bm25_index = (
        build_bm25_index(
            semantic_filtered_chunks
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metric_store = (
        initialize_metrics()
    )

    # ========================================================
    # Evaluate all questions
    # ========================================================

    print(
        "\nStarting evaluation..."
    )

    for question_number, item in enumerate(
        golden_dataset,
        start=1,
    ):

        question = item[
            "question"
        ]

        gold_passages = item[
            "gold_passages"
        ]

        print(
            f"\nQuestion "
            f"{question_number}/"
            f"{len(golden_dataset)}: "
            f"{question}"
        )

        # ====================================================
        # Recursive Vector
        # ====================================================

        recursive_vector_results, recursive_vector_time = (
            retrieve_vector(
                question,
                recursive_chunks,
                recursive_embeddings,
                embedding_model,
                VECTOR_CANDIDATE_K,
            )
        )

        # ====================================================
        # Recursive Vector + Metadata
        # ====================================================

        recursive_vector_metadata_results, recursive_vector_metadata_time = (
            retrieve_vector(
                question,
                recursive_filtered_chunks,
                recursive_filtered_embeddings,
                embedding_model,
                VECTOR_CANDIDATE_K,
            )
        )

        # ====================================================
        # Recursive BM25
        # ====================================================

        recursive_bm25_results, recursive_bm25_time = (
            retrieve_bm25(
                question,
                recursive_chunks,
                recursive_bm25_index,
                BM25_CANDIDATE_K,
            )
        )

        recursive_bm25_metadata_results, recursive_bm25_metadata_time = (
            retrieve_bm25(
                question,
                recursive_filtered_chunks,
                recursive_filtered_bm25_index,
                BM25_CANDIDATE_K,
            )
        )

        # ====================================================
        # Recursive Hybrid
        # ====================================================

        recursive_hybrid_results, recursive_rrf_time = (
            reciprocal_rank_fusion(
                recursive_vector_results,
                recursive_bm25_results,
            )
        )

        recursive_hybrid_metadata_results, recursive_metadata_rrf_time = (
            reciprocal_rank_fusion(
                recursive_vector_metadata_results,
                recursive_bm25_metadata_results,
            )
        )

        # ====================================================
        # Semantic Vector
        # ====================================================

        semantic_vector_results, semantic_vector_time = (
            retrieve_vector(
                question,
                semantic_chunks,
                semantic_embeddings,
                embedding_model,
                VECTOR_CANDIDATE_K,
            )
        )

        # ====================================================
        # Semantic Vector + Metadata
        # ====================================================

        semantic_vector_metadata_results, semantic_vector_metadata_time = (
            retrieve_vector(
                question,
                semantic_filtered_chunks,
                semantic_filtered_embeddings,
                embedding_model,
                VECTOR_CANDIDATE_K,
            )
        )

        # ====================================================
        # Semantic BM25
        # ====================================================

        semantic_bm25_results, semantic_bm25_time = (
            retrieve_bm25(
                question,
                semantic_chunks,
                semantic_bm25_index,
                BM25_CANDIDATE_K,
            )
        )

        semantic_bm25_metadata_results, semantic_bm25_metadata_time = (
            retrieve_bm25(
                question,
                semantic_filtered_chunks,
                semantic_filtered_bm25_index,
                BM25_CANDIDATE_K,
            )
        )

        # ====================================================
        # Semantic Hybrid
        # ====================================================

        semantic_hybrid_results, semantic_rrf_time = (
            reciprocal_rank_fusion(
                semantic_vector_results,
                semantic_bm25_results,
            )
        )

        semantic_hybrid_metadata_results, semantic_metadata_rrf_time = (
            reciprocal_rank_fusion(
                semantic_vector_metadata_results,
                semantic_bm25_metadata_results,
            )
        )

        # ====================================================
        # Configuration results
        # ====================================================

        configurations = {

            "Recursive Vector": (
                recursive_vector_results,
                recursive_vector_time,
                0.0,
                0.0,
            ),

            "Recursive Vector + Metadata": (
                recursive_vector_metadata_results,
                recursive_vector_metadata_time,
                0.0,
                0.0,
            ),

            "Recursive Hybrid": (
                recursive_hybrid_results,
                recursive_vector_time,
                recursive_bm25_time,
                recursive_rrf_time,
            ),

            "Recursive Hybrid + Metadata": (
                recursive_hybrid_metadata_results,
                recursive_vector_metadata_time,
                recursive_bm25_metadata_time,
                recursive_metadata_rrf_time,
            ),

            "Semantic Vector": (
                semantic_vector_results,
                semantic_vector_time,
                0.0,
                0.0,
            ),

            "Semantic Vector + Metadata": (
                semantic_vector_metadata_results,
                semantic_vector_metadata_time,
                0.0,
                0.0,
            ),

            "Semantic Hybrid": (
                semantic_hybrid_results,
                semantic_vector_time,
                semantic_bm25_time,
                semantic_rrf_time,
            ),

            "Semantic Hybrid + Metadata": (
                semantic_hybrid_metadata_results,
                semantic_vector_metadata_time,
                semantic_bm25_metadata_time,
                semantic_metadata_rrf_time,
            ),
        }

        # ====================================================
        # Calculate metrics
        # ====================================================

        for config_name, (
            results,
            vector_time,
            bm25_time,
            rrf_time,
        ) in configurations.items():

            metrics = calculate_metrics(
                results,
                gold_passages,
            )

            total_time = (
                vector_time
                + bm25_time
                + rrf_time
            )

            metrics[
                "vector_time_ms"
            ] = vector_time

            metrics[
                "bm25_time_ms"
            ] = bm25_time

            metrics[
                "rrf_time_ms"
            ] = rrf_time

            metrics[
                "total_retrieval_time_ms"
            ] = total_time

            add_metrics(
                metric_store,
                config_name,
                metrics,
            )

    # ========================================================
    # Final averages
    # ========================================================

    averages = average_metrics(
        metric_store
    )

    # ========================================================
    # Build final output
    # ========================================================

    output = build_results_table(
        averages
    )

    # ========================================================
    # Print results
    # ========================================================

    print(
        "\n"
        + output
    )

    # ========================================================
    # Save results
    # ========================================================

    save_results(
        output,
        len(recursive_chunks),
        len(recursive_filtered_chunks),
        len(semantic_chunks),
        len(semantic_filtered_chunks),
    )

    print(
        f"\nResults saved to:"
    )

    print(
        RESULTS_FILE
    )