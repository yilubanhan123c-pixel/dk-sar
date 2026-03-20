"""
向量数据库封装
使用 ChromaDB（本地向量库）+ sentence-transformers 做语义检索
"""
import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
import config


class VectorStore:
    """
    向量数据库，负责：
    1. 把 JSON 数据向量化并存储
    2. 根据输入文本检索最相似的内容
    """
    
    def __init__(self):
        print("📦 正在加载 Embedding 模型（首次使用会自动下载，请等待）...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # 初始化 ChromaDB（本地文件存储）
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
        
        self._positive_collection = None
        self._negative_collection = None
        print("✅ 向量数据库初始化完成")
    
    def _get_positive_collection(self):
        if self._positive_collection is None:
            self._positive_collection = self.client.get_or_create_collection("positive_cases")
        return self._positive_collection
    
    def _get_negative_collection(self):
        if self._negative_collection is None:
            self._negative_collection = self.client.get_or_create_collection("negative_fallacies")
        return self._negative_collection
    
    def build_index(self, force_rebuild: bool = False):
        """
        构建向量索引
        首次运行必须执行，后续运行会跳过（除非 force_rebuild=True）
        """
        pos_col = self._get_positive_collection()
        neg_col = self._get_negative_collection()
        
        # 如果已有数据且不强制重建，跳过
        if pos_col.count() > 0 and neg_col.count() > 0 and not force_rebuild:
            print(f"✅ 向量索引已存在（正样本: {pos_col.count()} 条，负样本: {neg_col.count()} 条），跳过构建")
            return
        
        # 构建正样本索引
        print("🔨 正在构建正样本库向量索引...")
        with open(config.POSITIVE_CASES_FILE, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        
        # 清除旧数据
        if force_rebuild and pos_col.count() > 0:
            self.client.delete_collection("positive_cases")
            pos_col = self.client.create_collection("positive_cases")
            self._positive_collection = pos_col
        
        documents = [case["embedding_text"] for case in cases]
        embeddings = self.model.encode(documents).tolist()
        ids = [case["case_id"] for case in cases]
        metadatas = [{"name": c["name"], "process_type": c["process_type"], 
                      "equipment": c["equipment"]} for c in cases]
        
        pos_col.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)
        print(f"   ✅ 正样本库索引完成（{len(cases)} 条案例）")
        
        # 构建负样本索引
        print("🔨 正在构建负样本库向量索引...")
        with open(config.NEGATIVE_FALLACIES_FILE, 'r', encoding='utf-8') as f:
            fallacies = json.load(f)
        
        if force_rebuild and neg_col.count() > 0:
            self.client.delete_collection("negative_fallacies")
            neg_col = self.client.create_collection("negative_fallacies")
            self._negative_collection = neg_col
        
        doc_neg = [f["embedding_text"] for f in fallacies]
        emb_neg = self.model.encode(doc_neg).tolist()
        ids_neg = [f["fallacy_id"] for f in fallacies]
        meta_neg = [{"category": f["category"], "false_claim": f["false_claim"][:100]} for f in fallacies]
        
        neg_col.add(documents=doc_neg, embeddings=emb_neg, ids=ids_neg, metadatas=meta_neg)
        print(f"   ✅ 负样本库索引完成（{len(fallacies)} 条谬误）")
    
    def search_similar_cases(self, query_text: str, top_k: int = None) -> list:
        """
        在正样本库中检索相似案例
        返回 top_k 个最相似的完整案例 JSON
        """
        scored = self.search_similar_cases_with_scores(query_text, top_k)
        return [item["case"] for item in scored]

    def search_similar_cases_with_scores(self, query_text: str, top_k: int = None) -> list:
        """
        在正样本库中检索相似案例，同时返回相似度分数
        返回: [{"case": {完整案例}, "similarity": 0~1}, ...]
        """
        if top_k is None:
            top_k = config.TOP_K_CASES

        pos_col = self._get_positive_collection()
        if pos_col.count() == 0:
            raise RuntimeError("正样本库为空！请先运行 build_index()")

        query_embedding = self.model.encode([query_text]).tolist()
        results = pos_col.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, pos_col.count()),
            include=["distances"],
        )

        # 加载完整案例数据
        with open(config.POSITIVE_CASES_FILE, 'r', encoding='utf-8') as f:
            all_cases = {case["case_id"]: case for case in json.load(f)}

        matched = []
        for i, case_id in enumerate(results["ids"][0]):
            if case_id in all_cases:
                distance = results["distances"][0][i]
                similarity = max(0, 1 - distance / 2)
                matched.append({"case": all_cases[case_id], "similarity": similarity})

        return matched
    
    def search_similar_fallacies(self, query_text: str, top_k: int = 5) -> list:
        """
        在负样本库中检索相似谬误
        返回相似度得分和谬误内容
        """
        neg_col = self._get_negative_collection()
        if neg_col.count() == 0:
            raise RuntimeError("负样本库为空！请先运行 build_index()")
        
        query_embedding = self.model.encode([query_text]).tolist()
        results = neg_col.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, neg_col.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        # 加载完整谬误数据
        with open(config.NEGATIVE_FALLACIES_FILE, 'r', encoding='utf-8') as f:
            all_fallacies = {f["fallacy_id"]: f for f in json.load(f)}
        
        matched = []
        for i, fallacy_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            # ChromaDB 返回的是 L2 距离，转换为相似度 (0~1)
            similarity = max(0, 1 - distance / 2)
            
            if fallacy_id in all_fallacies:
                matched.append({
                    "fallacy": all_fallacies[fallacy_id],
                    "similarity": similarity
                })
        
        return matched
    
    def encode_text(self, text: str) -> list:
        """将文本转换为向量"""
        return self.model.encode([text]).tolist()[0]


# 全局单例，避免重复加载模型
_vector_store_instance = None

def get_vector_store() -> VectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
