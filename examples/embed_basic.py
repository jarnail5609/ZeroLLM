"""Basic embedding example — convert text to vectors and compare similarity."""

from zerollm import Embed

emb = Embed()  # default: all-MiniLM-L6-v2

# Single text
vector = emb.encode("Hello world")
print(f"Dimension: {len(vector)}")

# Batch
vectors = emb.encode(["cats are great", "dogs are loyal", "python is fun"])
print(f"Batch: {len(vectors)} vectors")

# Similarity
score = emb.similarity("cats are great", "dogs are loyal")
print(f"cats vs dogs: {score:.3f}")

score = emb.similarity("cats are great", "python is fun")
print(f"cats vs python: {score:.3f}")
