"""BGE-M3 嵌入器：同时产出 dense(1024维) + sparse(词权重) 向量。"""
import numpy as np
from typing import List, Dict, Tuple


class BGEM3Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(model_name, use_fp16=(device != "cpu"), device=device)

    def embed_dense(self, texts: List[str]) -> np.ndarray:
        """返回 dense 向量，shape (n, 1024)。"""
        result = self.model.encode(texts, return_dense=True, return_sparse=False)
        return result["dense_vecs"]

    def embed_sparse(self, texts: List[str]) -> List[Dict[int, float]]:
        """返回稀疏词权重，每个文本对应一个 {word_id: weight} 字典。"""
        result = self.model.encode(texts, return_dense=False, return_sparse=True)
        return result["lexical_weights"]

    def embed_both(self, texts: List[str]) -> Tuple[np.ndarray, List[Dict[int, float]]]:
        """一次编码同时返回 dense 和 sparse。"""
        result = self.model.encode(texts, return_dense=True, return_sparse=True)
        return result["dense_vecs"], result["lexical_weights"]

    def embed_query(self, query: str) -> Tuple[np.ndarray, Dict[int, float]]:
        """对单条 query 编码。"""
        dense, sparse = self.embed_both([query])
        return dense[0], sparse[0]
