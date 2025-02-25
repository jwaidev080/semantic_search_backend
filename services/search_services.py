import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load model and tokenizer
try:
    model = AutoModel.from_pretrained("../embedding_models")
    tokenizer = AutoTokenizer.from_pretrained("../embedding_models")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    raise


def get_embeddings(texts: list[str]) -> list[list[float]]:
    try:
        # Tokenize inputs
        tokenized_inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # Generate embeddings
        with torch.no_grad():
            model_output = model(**tokenized_inputs)
            embeddings = model_output.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise


def semantic_search(query: str, documents: list[str], document_embeddings: list[list[float]], top_k: int = 3) -> list[tuple[str, float]]:
    try:
        query_embedding = get_embeddings([query])[0]
        similarities = cosine_similarity(
            [query_embedding], document_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(documents[i], float(similarities[i])) for i in top_indices]
        return results
    except Exception as e:
        print(f"Error performing semantic search: {e}")
        raise
