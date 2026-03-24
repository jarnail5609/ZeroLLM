"""Tests for the Embed class."""

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_sentence_transformers():
    """Mock sentence_transformers to avoid downloading models."""
    mock_st = MagicMock()
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384

    import numpy as np

    mock_model.encode.side_effect = lambda texts, **kw: np.random.rand(len(texts), 384).astype(
        np.float32
    )

    mock_st.SentenceTransformer.return_value = mock_model
    mock_st.util.cos_sim.return_value = [[0.85]]

    with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
        yield mock_model, mock_st


def test_embed_init(mock_sentence_transformers):
    from zerollm.embed import Embed

    emb = Embed("all-MiniLM-L6-v2")
    assert emb.model_name == "all-MiniLM-L6-v2"
    assert emb.dimension == 384


def test_encode_single(mock_sentence_transformers):
    from zerollm.embed import Embed

    emb = Embed()
    vector = emb.encode("Hello world")
    assert isinstance(vector, list)
    assert len(vector) == 384


def test_encode_batch(mock_sentence_transformers):
    from zerollm.embed import Embed

    emb = Embed()
    vectors = emb.encode(["Hello", "World"])
    assert isinstance(vectors, list)
    assert len(vectors) == 2
    assert len(vectors[0]) == 384


def test_similarity(mock_sentence_transformers):
    from zerollm.embed import Embed

    emb = Embed()
    score = emb.similarity("cat", "dog")
    assert isinstance(score, float)
    assert score == pytest.approx(0.85)
