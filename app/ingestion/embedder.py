"""BGE-M3 嵌入器：同时产出 dense(1024维) + sparse(词权重) 向量。"""
import os
import numpy as np
from typing import List, Dict, Tuple

from milvus_model.hybrid import BGEM3EmbeddingFunction


class BGEM3Embedder:
    def __init__(self, model_name: str = None, device: str = "cpu"):
        if model_name is None:
            from app.core.config import config
            model_name = os.path.join(config.MODELS_DIR, "bge-m3")
        self.ef = BGEM3EmbeddingFunction(model_name=model_name, device=device)

    def embed_dense(self, texts: List[str]) -> np.ndarray:
        """返回 dense 向量，shape (n, 1024)。"""
        result = self.ef.encode_documents(texts)
        return np.array(result["dense"])

    def embed_sparse(self, texts: List[str]) -> List[Dict[int, float]]:
        """返回稀疏词权重，每个文本对应一个 {word_id: weight} 字典。"""
        result = self.ef.encode_documents(texts)
        return _sparse_to_dicts(result["sparse"])

    def embed_both(self, texts: List[str]) -> Tuple[np.ndarray, List[Dict[int, float]]]:
        """一次编码同时返回 dense 和 sparse。"""
        result = self.ef.encode_documents(texts)
        return np.array(result["dense"]), _sparse_to_dicts(result["sparse"])

    def embed_query(self, query: str) -> Tuple[np.ndarray, Dict[int, float]]:
        """对单条 query 编码。"""
        result = self.ef.encode_queries([query])
        return np.array(result["dense"][0]), _sparse_row_to_dict(result["sparse"], 0)


def _sparse_to_dicts(sparse) -> List[Dict[int, float]]:
    """scipy sparse matrix → [{word_id: weight}, ...] (Milvus sparse format)。"""
    sparse = sparse.tocsr()
    dicts = []
    for i in range(sparse.shape[0]):
        row = sparse[i]
        if row.nnz > 0:
            cols = row.col if hasattr(row, 'col') else row.indices
            dicts.append({int(j): float(v) for j, v in zip(cols, row.data)})
        else:
            dicts.append({})
    return dicts


def _sparse_row_to_dict(sparse, row_idx: int) -> Dict[int, float]:
    """scipy sparse matrix 单行 → {word_id: weight}。"""
    sparse = sparse.tocsr()
    row = sparse[row_idx]
    if row.nnz > 0:
        cols = row.col if hasattr(row, 'col') else row.indices
        return {int(j): float(v) for j, v in zip(cols, row.data)}
    return {}
