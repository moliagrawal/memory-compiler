"""Unit tests for FAISS indexing: Flat, IVF-PQ, and HNSW."""

import numpy as np
import pytest
from mc.index.faiss_index import VideoFAISSIndex


def test_faiss_index_types():
    N = 100
    D = 512
    rng = np.random.RandomState(42)
    features = rng.randn(N, D).astype(np.float32)
    query = rng.randn(D).astype(np.float32)

    for itype in ["Flat", "IVF-PQ", "HNSW"]:
        idx = VideoFAISSIndex(d_model=D, index_type=itype, nlist=8, m=16)
        stats = idx.build(features)

        assert stats["num_indexed"] == N
        assert stats["build_time_ms"] > 0

        scores, indices, latency_ms = idx.search(query, top_k=5)
        assert len(scores) == 5
        assert len(indices) == 5
        assert latency_ms >= 0.0
