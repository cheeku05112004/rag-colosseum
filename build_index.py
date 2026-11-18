import re
import numpy as np
from fastembed import TextEmbedding
from knowledge_base import PASSAGES

def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)

SENTENCES = []

for passage in PASSAGES:
    parts = re.split(r'(?<=[.!?])\s+', passage.strip())
    SENTENCES.extend(sentence for sentence in parts if sentence)

print("Loading embedding model...")
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

print("Generating passage embeddings...")
passage_vectors = normalize(
    np.array(list(model.embed(PASSAGES)), dtype=np.float32)
)

print("Generating sentence embeddings...")
sentence_vectors = normalize(
    np.array(list(model.embed(SENTENCES)), dtype=np.float32)
)

np.save("embeddings.npy", passage_vectors)
np.save("sentence_embeddings.npy", sentence_vectors)

print("INDEX CREATED SUCCESSFULLY!")
print("Passages:", len(PASSAGES))
print("Sentences:", len(SENTENCES))
print("Passage shape:", passage_vectors.shape)
print("Sentence shape:", sentence_vectors.shape)
