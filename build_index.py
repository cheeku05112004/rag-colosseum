import numpy as np
from fastembed import TextEmbedding
from knowledge_base import PASSAGES

print("Loading embedding model...")
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

print("Generating embeddings for 50 passages...")
vectors = np.array(
    list(model.embed(PASSAGES)),
    dtype=np.float32
)

# Normalize vectors for cosine similarity
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
vectors = vectors / np.maximum(norms, 1e-12)

np.save("embeddings.npy", vectors)

print("\nINDEX CREATED SUCCESSFULLY!")
print("Passages:", len(PASSAGES))
print("Embedding shape:", vectors.shape)
print("Saved file: embeddings.npy")
