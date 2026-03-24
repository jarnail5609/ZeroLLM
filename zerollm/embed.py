"""Embed — standalone embedding API for text to vectors."""

from __future__ import annotations

from rich.console import Console

console = Console()


class Embed:
    """Generate embeddings from text using sentence-transformers.

    Usage:
        from zerollm import Embed

        emb = Embed()
        vector = emb.encode("Hello world")
        vectors = emb.encode(["Hello", "World"])
        similarity = emb.similarity("cat", "dog")
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        """Initialize Embed.

        Args:
            model: Sentence transformer model name from HuggingFace.
        """
        from sentence_transformers import SentenceTransformer

        console.print(f"[dim]Loading embedding model ({model})...[/dim]")
        self._model = SentenceTransformer(model)
        self.model_name = model
        self.dimension = self._model.get_sentence_embedding_dimension()
        console.print(f"[green]✓[/green] Embed ready ({model}, dim={self.dimension})")

    def encode(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Convert text to embedding vector(s).

        Args:
            text: A single string or list of strings.

        Returns:
            A single vector (list of floats) for a string input,
            or a list of vectors for a list input.
        """
        single = isinstance(text, str)
        inputs = [text] if single else text
        embeddings = self._model.encode(inputs, show_progress_bar=False)
        results = [emb.tolist() for emb in embeddings]
        return results[0] if single else results

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Cosine similarity score between -1.0 and 1.0.
        """
        from sentence_transformers import util

        emb_a = self._model.encode([text_a])
        emb_b = self._model.encode([text_b])
        score = util.cos_sim(emb_a, emb_b)
        return float(score[0][0])
