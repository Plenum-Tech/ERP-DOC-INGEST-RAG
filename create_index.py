# create_index.py
from __future__ import annotations
import os
import sys
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)

# Try semantic imports (SDK versions vary)
try:
    from azure.search.documents.indexes.models import (  # type: ignore
        SemanticSettings,
        SemanticConfiguration,
        PrioritizedFields,
        SemanticField,
    )
    SEMANTIC_AVAILABLE = True
except Exception:
    SEMANTIC_AVAILABLE = False

load_dotenv()

ENDPOINT      = os.environ["AZURE_SEARCH_ENDPOINT"]
KEY           = os.environ["AZURE_SEARCH_KEY"]
INDEX         = os.getenv("AZURE_SEARCH_INDEX", "hr-policies")
EMBED_MODEL   = os.getenv("EMBED_MODEL", "text-embedding-3-small")
ENABLE_SEM    = os.getenv("ENABLE_SEMANTIC", "1").lower() in {"1", "true", "yes"}
FORCE_RECREATE= os.getenv("FORCE_RECREATE", "0").lower() in {"1", "true", "yes"}

# ----- Dimensions: keep in sync with your embedding model -----
def dims_for_model(model: str) -> int:
    m = (model or "").lower()
    # OpenAI text-embedding-3 family
    if "text-embedding-3-large" in m:
        return 3072
    if "text-embedding-3-small" in m:
        return 1536
    # Fallbacks / common older models
    if "text-embedding-ada-002" in m:
        return 1536
    # Allow manual override via EMBED_DIMS
    return int(os.getenv("EMBED_DIMS", "1536"))

DIMS = dims_for_model(EMBED_MODEL)

# ----- Build index definition -----
def build_index(include_semantics: bool) -> SearchIndex:
    """
    Fields are designed to work with the DI-driven chunkers:
      - 'content' holds clean text (no overlap)
      - 'page' is the primary page for the chunk (min page in a span)
      - Optional fields allow future expansion without breaking ingestion
    """
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchField(name="title", type=SearchFieldDataType.String, searchable=True, filterable=True, sortable=True, facetable=True),
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
        # Optional/forward-compatible metadata (ingestor can start populating later)
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SimpleField(name="page_first", type=SearchFieldDataType.Int32, filterable=True, sortable=True, retrievable=False),
        SimpleField(name="page_last", type=SearchFieldDataType.Int32, filterable=True, sortable=True, retrievable=False),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SimpleField(name="doc_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchField(name="section_path", type=SearchFieldDataType.String, searchable=True, filterable=True),
        SimpleField(name="is_table", type=SearchFieldDataType.Boolean, filterable=True),
        # Vector field
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=DIMS,
            vector_search_profile_name="v1",
        ),
    ]

    # HNSW configuration tuned for most RAG workloads
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-1",
                parameters={
                    "m": 16,               # graph degree
                    "efConstruction": 400, # build quality
                    "efSearch": 200,       # query recall/speed tradeoff
                    "metric": "cosine",
                },
            )
        ],
        profiles=[VectorSearchProfile(name="v1", algorithm_configuration_name="hnsw-1")],
    )

    kwargs = {}
    if include_semantics:
        kwargs["semantic_settings"] = SemanticSettings(
            configurations=[
                SemanticConfiguration(
                    name="semantic-hr",
                    prioritized_fields=PrioritizedFields(
                        title_field=SemanticField(field_name="title"),
                        content_fields=[SemanticField(field_name="content"), SemanticField(field_name="section_path")],
                    ),
                )
            ]
        )

    return SearchIndex(name=INDEX, fields=fields, vector_search=vector_search, **kwargs)

# ----- Create or update with dimension safety -----
def main():
    client = SearchIndexClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))

    def ensure_index(include_semantics: bool):
        idx_def = build_index(include_semantics=include_semantics and SEMANTIC_AVAILABLE)
        return client.create_or_update_index(idx_def)

    try:
        existing = client.get_index(INDEX)
        # Validate vector dims; Azure Search does not allow changing vector_search_dimensions in-place
        try:
            vec_field = next(f for f in existing.fields if f.name == "embedding")
            existing_dims = getattr(vec_field, "vector_search_dimensions", None)
        except StopIteration:
            existing_dims = None

        if existing_dims and int(existing_dims) != int(DIMS):
            msg = (
                f"[!] Index '{INDEX}' exists with embedding dims={existing_dims}, "
                f"but your model expects dims={DIMS}."
            )
            if not FORCE_RECREATE:
                print(msg)
                print("    Set FORCE_RECREATE=1 to drop & recreate the index, or update EMBED_MODEL/EMBED_DIMS.")
                sys.exit(1)
            else:
                print(msg + " Dropping & recreating (FORCE_RECREATE=1).")
                client.delete_index(INDEX)
                ensure_index(include_semantics=ENABLE_SEM)
                print(f"[OK] Recreated index '{INDEX}' (dims={DIMS}, semantic={'on' if (ENABLE_SEM and SEMANTIC_AVAILABLE) else 'off'}).")
                return

        # Safe to update in-place (adds fields, updates HNSW/profile, attaches semantic if available)
        ensure_index(include_semantics=ENABLE_SEM)
        print(f"[OK] Updated index '{INDEX}' (dims={DIMS}, semantic={'on' if (ENABLE_SEM and SEMANTIC_AVAILABLE) else 'off'}).")

    except Exception:
        # Not found → create
        ensure_index(include_semantics=ENABLE_SEM)
        print(f"[OK] Created index '{INDEX}' (dims={DIMS}, semantic={'on' if (ENABLE_SEM and SEMANTIC_AVAILABLE) else 'off'}).")

if __name__ == "__main__":
    main()
