# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.4: Embeddings & Vector Search
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Understanding embeddings
# MAGIC - Different embedding models
# MAGIC - Creating vector search endpoints
# MAGIC - Creating and syncing vector search indexes
# MAGIC - Querying with similarity search
# MAGIC - Advanced options: filters, hybrid search, reranking

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.reranker import DatabricksReranker

from arxiv_curator.config import load_config, get_env
from arxiv_curator.vector_search import VectorSearchManager

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)
catalog = cfg.catalog
schema = cfg.schema

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understanding Embeddings
# MAGIC
# MAGIC **Embeddings** are numerical representations of text that capture semantic meaning.
# MAGIC
# MAGIC ### Key Concepts:
# MAGIC
# MAGIC - **Vector**: Array of numbers (e.g., [0.1, -0.3, 0.5, ...])
# MAGIC - **Dimension**: Length of the vector (e.g., 384, 768, 1024)
# MAGIC - **Semantic Similarity**: Similar meanings = similar vectors
# MAGIC - **Distance Metrics**: Cosine similarity, Euclidean distance, dot product
# MAGIC
# MAGIC ### How it Works:
# MAGIC
# MAGIC ```
# MAGIC Text: "machine learning"
# MAGIC   ↓ (Embedding Model)
# MAGIC Vector: [0.23, -0.15, 0.67, ..., 0.42]  # 1024 dimensions
# MAGIC
# MAGIC Text: "artificial intelligence"
# MAGIC   ↓ (Embedding Model)
# MAGIC Vector: [0.25, -0.13, 0.65, ..., 0.40]  # Similar to above!
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Embedding Models Comparison
# MAGIC
# MAGIC | Model | Dimensions | Max Tokens | Best For |
# MAGIC |-------|-----------|------------|----------|
# MAGIC | **databricks-bge-large-en** | 1024 | 512 | General purpose, high quality |
# MAGIC | **databricks-gte-large-en** | 1024 | 512 | General purpose, fast |
# MAGIC | **text-embedding-ada-002** (OpenAI) | 1536 | 8191 | High quality, expensive |
# MAGIC | **e5-large-v2** | 1024 | 512 | Open source, good quality |
# MAGIC | **all-MiniLM-L6-v2** | 384 | 512 | Fast, smaller, lower quality |
# MAGIC
# MAGIC **For this course, we'll use `databricks-gte-large-en`** - it's fast, high-quality, and free on Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Vector Search Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────┐
# MAGIC │     Delta Table (arxiv_chunks)          │
# MAGIC │  - id                                    │
# MAGIC │  - text                                  │
# MAGIC │  - metadata (title, author, etc.)       │
# MAGIC └──────────────┬──────────────────────────┘
# MAGIC                │
# MAGIC                │ (Automatic sync)
# MAGIC                ↓
# MAGIC ┌─────────────────────────────────────────┐
# MAGIC │     Vector Search Index                  │
# MAGIC │  - Embeddings generated automatically    │
# MAGIC │  - Stored in optimized format            │
# MAGIC │  - Supports similarity search            │
# MAGIC └──────────────┬──────────────────────────┘
# MAGIC                │
# MAGIC                │ (Query)
# MAGIC                ↓
# MAGIC ┌─────────────────────────────────────────┐
# MAGIC │     Search Results                       │
# MAGIC │  - Most similar chunks                   │
# MAGIC │  - With similarity scores                │
# MAGIC └─────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Vector Search Endpoint

# COMMAND ----------

# Using VectorSearchManager from arxiv_curator.vector_search
# This handles endpoint and index creation automatically

vs_manager = VectorSearchManager(
    config=cfg,
    endpoint_name=cfg.vector_search_endpoint,
    embedding_model=cfg.embedding_endpoint
)

logger.info(f"Vector Search Endpoint: {vs_manager.endpoint_name}")
logger.info(f"Embedding Model: {vs_manager.embedding_model}")
logger.info(f"Index Name: {vs_manager.index_name}")

# COMMAND ----------

# Create endpoint if it doesn't exist
vs_manager.create_endpoint_if_not_exists()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Endpoint Types:
# MAGIC
# MAGIC - **STANDARD**: General purpose, good performance
# MAGIC - **STANDARD_LARGE**: Higher throughput, more expensive
# MAGIC
# MAGIC For development and most production workloads, STANDARD is sufficient.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create Vector Search Index

# COMMAND ----------

# Create or get the vector search index using VectorSearchManager
# This automatically:
# - Creates the index if it doesn't exist
# - Configures it with the embedding model
# - Sets up delta sync with the arxiv_chunks table

index = vs_manager.create_or_get_index()

logger.info(f"\n✓ Vector search setup complete!")
logger.info(f"  Index: {vs_manager.index_name}")
logger.info(f"  Source: {vs_manager.catalog}.{vs_manager.schema}.arxiv_chunks")
logger.info(f"  Embedding Model: {vs_manager.embedding_model}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Index Configuration Options:
# MAGIC
# MAGIC - **pipeline_type**:
# MAGIC   - `TRIGGERED`: Manual sync, good for batch processing
# MAGIC   - `CONTINUOUS`: Auto-sync with Change Data Feed, real-time updates
# MAGIC
# MAGIC - **primary_key**: Unique identifier for each document
# MAGIC
# MAGIC - **embedding_source_column**: The text column to embed
# MAGIC
# MAGIC - **embedding_model_endpoint_name**: Which embedding model to use

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Helper Function for Parsing Results

# COMMAND ----------

def parse_vector_search_results(results):
    """Parse vector search results from array format to dict format.
    
    Args:
        results: Raw results from similarity_search()
        
    Returns:
        List of dictionaries with column names as keys
    """
    columns = [col['name'] for col in results.get('manifest', {}).get('columns', [])]
    data_array = results.get('result', {}).get('data_array', [])
    
    return [dict(zip(columns, row_data)) for row_data in data_array]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Semantic Search with Similarity
# MAGIC
# MAGIC ### How Semantic Search Works
# MAGIC
# MAGIC 1. **Query Embedding**: Convert your search query to a vector
# MAGIC 2. **Similarity Calculation**: Compare query vector to all document vectors using **cosine similarity**
# MAGIC 3. **Ranking**: Return documents with highest similarity scores
# MAGIC
# MAGIC ### Cosine Similarity
# MAGIC
# MAGIC Measures the angle between two vectors (range: -1 to 1):
# MAGIC - **1.0**: Identical meaning
# MAGIC - **0.8-0.9**: Very similar
# MAGIC - **0.5-0.7**: Somewhat related
# MAGIC - **< 0.5**: Less relevant
# MAGIC
# MAGIC ```
# MAGIC Query: "machine learning techniques"
# MAGIC   ↓ (Embedding)
# MAGIC Vector: [0.2, 0.5, -0.1, ...]
# MAGIC   ↓ (Cosine similarity with all docs)
# MAGIC Results ranked by similarity score
# MAGIC ```

# COMMAND ----------

# Simple similarity search
query = "What are the latest techniques in machine learning?"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title", "arxiv_id"],
    num_results=5
)

logger.info(f"Query: {query}\n")
logger.info("Top 5 Results:")
logger.info("=" * 80)

# Parse results using helper function
for i, row in enumerate(parse_vector_search_results(results), 1):
    logger.info(f"\n{i}. Paper: {row.get('title', 'N/A')}")
    logger.info(f"   arXiv ID: {row.get('arxiv_id', 'N/A')}")
    logger.info(f"   Chunk ID: {row.get('id', 'N/A')}")
    logger.info(f"   Text preview: {row.get('text', '')[:200]}...")
    logger.info(f"   Score: {row.get('score', 'N/A'):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Advanced Search: Filters

# COMMAND ----------

# Search with metadata filters
query = "neural networks and deep learning"

# Filter for papers from 2024 or later
results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title", "year", "authors"],
    filters={"year": "2026"},  # Only papers from 2024
    num_results=3
)

logger.info(f"Query: {query}")
logger.info(f"Filter: year = 2026\n")
logger.info("Results:")
logger.info("=" * 80)

for i, row in enumerate(parse_vector_search_results(results), 1):
    logger.info(f"\n{i}. {row.get('title', 'N/A')}")
    logger.info(f"   Year: {row.get('year', 'N/A')}")
    authors = row.get('authors', 'N/A')
    logger.info(f"   Authors: {str(authors)[:100]}...")
    logger.info(f"   Text: {row.get('text', '')[:150]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filter Examples:
# MAGIC
# MAGIC ```python
# MAGIC # Single filter
# MAGIC filters = {"year": "2024"}
# MAGIC
# MAGIC # Multiple filters (AND)
# MAGIC filters = {"year": "2024", "month": "01"}
# MAGIC
# MAGIC # Range filter
# MAGIC filters = {"year >= 2023"}
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Hybrid Search: Semantic + Keyword
# MAGIC
# MAGIC ### Why Hybrid Search?
# MAGIC
# MAGIC **Semantic search alone** may miss:
# MAGIC - Exact technical terms (e.g., "GPT-4" vs "language model")
# MAGIC - Acronyms and abbreviations
# MAGIC - Specific product names or codes
# MAGIC
# MAGIC **Hybrid search** combines:
# MAGIC - **Semantic search** (embeddings) → Captures meaning, synonyms
# MAGIC - **Keyword search** (BM25) → Exact term matching, TF-IDF scoring
# MAGIC
# MAGIC ### How It Works
# MAGIC
# MAGIC 1. Run both searches in parallel
# MAGIC 2. Get top-k results from each
# MAGIC 3. **Fusion**: Merge and rerank using:
# MAGIC    - Reciprocal Rank Fusion (RRF)
# MAGIC    - Weighted score combination
# MAGIC 4. Return final top-k
# MAGIC
# MAGIC ### BM25 (Best Match 25)
# MAGIC
# MAGIC Keyword scoring algorithm that considers:
# MAGIC - **Term frequency**: How often does the term appear?
# MAGIC - **Document length**: Normalize by doc length
# MAGIC - **Inverse document frequency**: Rare terms = higher weight
# MAGIC
# MAGIC **Result**: Better precision on technical queries with specific terminology.

# COMMAND ----------

# Hybrid search example
query = "transformer architecture attention mechanism"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title"],
    num_results=5,
    query_type="hybrid"  # Enable hybrid search
)

logger.info(f"Query: {query}")
logger.info("Search Type: Hybrid (Semantic + Keyword)\n")
logger.info("Results:")
logger.info("=" * 80)

for i, row in enumerate(parse_vector_search_results(results), 1):
    logger.info(f"\n{i}. {row.get('title', 'N/A')}")
    logger.info(f"   Text: {row.get('text', '')[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Reranking for Higher Precision
# MAGIC
# MAGIC ### The Two-Stage Retrieval Pattern
# MAGIC
# MAGIC **Stage 1: Fast Retrieval** (Bi-encoder)
# MAGIC - Retrieve top 20-50 candidates quickly
# MAGIC - Uses pre-computed embeddings
# MAGIC - Fast but less accurate
# MAGIC
# MAGIC **Stage 2: Precise Reranking** (Cross-encoder)
# MAGIC - Score each candidate against the query
# MAGIC - More accurate relevance scoring
# MAGIC - Slower, but only runs on candidates
# MAGIC
# MAGIC ### Bi-encoder vs Cross-encoder
# MAGIC
# MAGIC | Aspect | Bi-encoder | Cross-encoder |
# MAGIC |--------|-----------|---------------|
# MAGIC | **Speed** | Very fast | Slower |
# MAGIC | **Accuracy** | Good | Excellent |
# MAGIC | **Use case** | Initial retrieval | Reranking |
# MAGIC | **How it works** | Separate query & doc embeddings | Joint query-doc encoding |
# MAGIC
# MAGIC ### When to Use Reranking
# MAGIC
# MAGIC - **High-stakes queries**: Customer support, legal, medical
# MAGIC - **Complex queries**: Multi-faceted questions
# MAGIC - **When precision matters more than speed**
# MAGIC
# MAGIC ### Trade-offs
# MAGIC
# MAGIC - **Pros**: 10-30% improvement in relevance
# MAGIC - **Cons**: 2-5x slower, higher compute cost

# COMMAND ----------

# Search with reranking
query = "large language models for code generation"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title", "summary"],
    num_results=5,
    query_type="hybrid",
    reranker=DatabricksReranker(
        columns_to_rerank=["text", "title", "summary"]
    )
)

logger.info(f"Query: {query}")
logger.info("With reranking on: text, title, summary\n")
logger.info("Results:")
logger.info("=" * 80)

for i, row in enumerate(parse_vector_search_results(results), 1):
    logger.info(f"\n{i}. {row.get('title', 'N/A')}")
    logger.info(f"   Summary: {row.get('summary', '')[:150]}...")
    logger.info(f"   Text: {row.get('text', '')[:150]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Search Quality Comparison

# COMMAND ----------

# Compare different search strategies
query = "attention mechanisms in transformers"

logger.info(f"Query: {query}\n")

# Strategy 1: Basic semantic search
results_basic = index.similarity_search(
    query_text=query,
    columns=["text", "title"],
    num_results=3
)

logger.info("Strategy 1: Basic Semantic Search")
logger.info("-" * 80)
for i, row in enumerate(parse_vector_search_results(results_basic), 1):
    logger.info(f"{i}. {row.get('title', 'N/A')[:60]}...")

# Strategy 2: Hybrid search
results_hybrid = index.similarity_search(
    query_text=query,
    columns=["text", "title"],
    num_results=3,
    query_type="hybrid"
)

logger.info("\nStrategy 2: Hybrid Search")
logger.info("-" * 80)
for i, row in enumerate(parse_vector_search_results(results_hybrid), 1):
    logger.info(f"{i}. {row.get('title', 'N/A')[:60]}...")

# Strategy 3: Hybrid + Reranking
results_reranked = index.similarity_search(
    query_text=query,
    columns=["text", "title"],
    num_results=3,
    query_type="hybrid",
    reranker=DatabricksReranker(columns_to_rerank=["text", "title"])
)

logger.info("\nStrategy 3: Hybrid + Reranking")
logger.info("-" * 80)
for i, row in enumerate(parse_vector_search_results(results_reranked), 1):
    logger.info(f"{i}. {row.get('title', 'N/A')[:60]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Best Practices
# MAGIC
# MAGIC ### ✅ Do:
# MAGIC 1. **Use hybrid search** for better recall
# MAGIC 2. **Add reranking** for critical applications
# MAGIC 3. **Filter by metadata** to narrow results
# MAGIC 4. **Monitor index sync** status
# MAGIC 5. **Use appropriate num_results** (5-10 for most cases)
# MAGIC 6. **Include relevant columns** in results
# MAGIC 7. **Test different embedding models** for your use case
# MAGIC
# MAGIC ### ❌ Don't:
# MAGIC 1. Retrieve too many results (increases latency)
# MAGIC 2. Ignore index sync status
# MAGIC 3. Use semantic search for exact keyword matches
# MAGIC 4. Forget to handle empty results
# MAGIC 5. Over-rely on similarity scores alone

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Monitoring and Maintenance

# COMMAND ----------

# Check index status
index_info = vs_manager.client.get_index(
    endpoint_name=vs_manager.endpoint_name,
    index_name=vs_manager.index_name
)

logger.info("Index Information:")
logger.info(f"  Name: {index_info.name}")
logger.info(f"  Endpoint: {index_info.endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Index Maintenance:
# MAGIC
# MAGIC ```python
# MAGIC # Sync index manually (for TRIGGERED pipeline)
# MAGIC index.sync()
# MAGIC
# MAGIC # Delete index (if needed)
# MAGIC # vs_manager.client.delete_index(index_name=vs_manager.index_name)
# MAGIC
# MAGIC # Update index (change configuration)
# MAGIC # Requires recreation in most cases
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, we learned:
# MAGIC
# MAGIC 1. ✅ Understanding embeddings and vector representations
# MAGIC 2. ✅ Comparing different embedding models
# MAGIC 3. ✅ Creating vector search endpoints
# MAGIC 4. ✅ Creating and syncing vector search indexes
# MAGIC 5. ✅ Basic similarity search
# MAGIC 6. ✅ Advanced features: filters, hybrid search, reranking
# MAGIC 7. ✅ Comparing search strategies
# MAGIC 8. ✅ Best practices and monitoring
# MAGIC
# MAGIC **Next**: Lecture 2.5 - Pipeline Design & Workflow
