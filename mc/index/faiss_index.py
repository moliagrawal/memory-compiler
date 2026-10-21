"""FAISS indexing support: Exact Flat, IVF-PQ, and HNSW (Phase 8)."""

import time
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class VideoFAISSIndex:
    """
    Multimodal frame indexer comparing:
      - Flat: Exact cosine similarity search
      - IVF-PQ: Inverted file index with Product Quantization for memory reduction
      - HNSW: Hierarchical Navigable Small World graph for sub-millisecond retrieval
    """

    def __init__(
        self,
        d_model: int = 512,
        index_type: str = "Flat",
        nlist: int = 16,
        m: int = 16,
        nbits: int = 8
    ):
        self.d_model = d_model
        self.index_type = index_type.upper()
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.index = None
        self.frame_to_meta = []
        self.build_time_s = 0.0

    def build(self, frame_features: np.ndarray, metadata_list: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Build and train index on frame features [N, D].
        """
        N, D = frame_features.shape
        assert D == self.d_model, f"Dimension mismatch: expected {self.d_model}, got {D}"

        # Normalize for cosine similarity via inner product
        feats_norm = frame_features.astype(np.float32)
        norms = np.linalg.norm(feats_norm, axis=-1, keepdims=True) + 1e-8
        feats_norm = feats_norm / norms

        start_t = time.perf_counter()

        if FAISS_AVAILABLE:
            if self.index_type == "FLAT":
                self.index = faiss.IndexFlatIP(self.d_model)
                self.index.add(feats_norm)

            elif self.index_type == "IVF-PQ" or self.index_type == "IVFPQ":
                quantizer = faiss.IndexFlatIP(self.d_model)
                # Adjust nlist and nbits if N is small
                effective_nlist = min(self.nlist, max(1, N // 4))
                effective_m = min(self.m, self.d_model)
                while self.d_model % effective_m != 0 and effective_m > 1:
                    effective_m -= 1

                # 8-bit sub-quantizer requires >= 256 points for clustering
                effective_nbits = 4 if N < 256 else self.nbits

                self.index = faiss.IndexIVFPQ(
                    quantizer, self.d_model, effective_nlist, effective_m, effective_nbits, faiss.METRIC_INNER_PRODUCT
                )
                if not self.index.is_trained:
                    self.index.train(feats_norm)
                self.index.add(feats_norm)
                self.index.nprobe = min(4, effective_nlist)

            elif self.index_type == "HNSW":
                self.index = faiss.IndexHNSWFlat(self.d_model, 32, faiss.METRIC_INNER_PRODUCT)
                self.index.hnsw.efSearch = 64
                self.index.add(feats_norm)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
        else:
            # Fallback pure-numpy flat index
            self.index = ("numpy_flat", feats_norm)

        self.build_time_s = time.perf_counter() - start_t
        self.frame_to_meta = metadata_list or []

        # Approximate index size
        if FAISS_AVAILABLE and hasattr(self.index, "ntotal"):
            # Estimate bytes
            if self.index_type == "FLAT":
                size_bytes = N * D * 4
            elif "IVF" in self.index_type:
                size_bytes = N * self.m + self.nlist * D * 4
            else:
                size_bytes = N * D * 4 + N * 32 * 8
        else:
            size_bytes = N * D * 4

        return {
            "num_indexed": N,
            "index_type": self.index_type,
            "build_time_ms": round(self.build_time_s * 1000, 2),
            "estimated_size_kb": round(size_bytes / 1024, 2)
        }

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Search for top-K matching frames.
        Returns:
            scores: [top_k] similarity scores
            indices: [top_k] matching frame indices
            latency_ms: search time in milliseconds
        """
        q_norm = query_vector.astype(np.float32).reshape(1, -1)
        q_norm = q_norm / (np.linalg.norm(q_norm) + 1e-8)

        start_t = time.perf_counter()

        if FAISS_AVAILABLE and not isinstance(self.index, tuple):
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(q_norm, k)
            latency_ms = (time.perf_counter() - start_t) * 1000
            return scores[0], indices[0], latency_ms
        else:
            # Numpy search fallback
            feats_norm = self.index[1] if isinstance(self.index, tuple) else None
            if feats_norm is None:
                return np.array([]), np.array([]), 0.0
            sims = np.dot(feats_norm, q_norm.T).flatten()
            k = min(top_k, len(sims))
            top_idx = np.argsort(-sims)[:k]
            latency_ms = (time.perf_counter() - start_t) * 1000
            return sims[top_idx], top_idx, latency_ms
