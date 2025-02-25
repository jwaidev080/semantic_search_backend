import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Path to the model (replace with the actual path or Hugging Face model name)
model_path = "sentence-transformers/all-MiniLM-L6-v2"

# Load the model and tokenizer
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode


def get_embeddings(texts: list[str]) -> list[list[float]]:
    # Tokenize inputs
    tokenized_inputs = tokenizer(
        texts,
        padding=True,  # Pad to the longest sequence in the batch
        truncation=True,  # Truncate to the model's max length
        return_tensors='pt'  # Return PyTorch tensors
    )

    # Generate embeddings
    with torch.no_grad():  # Disable gradient calculation for inference
        # Forward pass through the model
        model_output = model(**tokenized_inputs)

        # Perform CLS pooling (use the [CLS] token's embedding)
        embeddings = model_output.last_hidden_state[:, 0, :]

        # Normalize the embeddings (L2 normalization)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.tolist()


def semantic_search(query: str, documents: list[str], document_embeddings: list[list[float]], top_k: int = 3) -> list[tuple[str, float]]:
    # Get the embedding for the query
    query_embedding = get_embeddings([query])[0]

    # Compute cosine similarity between the query and document embeddings
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]

    # Get the top-k most similar documents
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = [(documents[i], float(similarities[i])) for i in top_indices]

    return results
