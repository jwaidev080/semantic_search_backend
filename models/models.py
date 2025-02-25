from pydantic import BaseModel


class QueryRequest(BaseModel):
    queries: list[str]


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]


class SearchRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    document: str
    similarity: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
