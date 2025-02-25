from fastapi import APIRouter, HTTPException
from models.models import QueryRequest, EmbeddingResponse, SearchRequest, SearchResponse, SearchResult
from services.search_services import get_embeddings, semantic_search

router = APIRouter()

# Example documents
documents = [
    "My achy breaky heart is a song by Billy Ray Cyrus.",
    "A summit is the highest point of a mountain or hill.",
    "Billy Ray Cyrus is a country music singer.",
    "The Eiffel Tower is located in Paris, France.",
    "Mount Everest is the highest peak in the world."
]

# Pre-compute document embeddings
document_embeddings = get_embeddings(documents)

# Perform embeddings


@router.post("/embeddings/", response_model=EmbeddingResponse)
async def get_query_embeddings(query_request: QueryRequest):
    try:
        embeddings = get_embeddings(query_request.queries)
        return EmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Perform semantic search
@router.post("/search/", response_model=SearchResponse)
async def search_documents(search_request: SearchRequest):
    try:
        results = semantic_search(
            search_request.query, documents, document_embeddings)

        formatted_results = [SearchResult(
            document=doc, similarity=sim) for doc, sim in results]
        return SearchResponse(results=formatted_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def read_root():
    return {"message": "Welcome to the Semantic Search API, This is test task!"}
